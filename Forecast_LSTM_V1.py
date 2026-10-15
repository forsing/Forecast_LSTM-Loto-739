#!/usr/bin/env python3

# numpy 2.4.6


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
import gc
import importlib.metadata
from pathlib import Path

import numpy as np


DEFAULTS = [
    "/data/loto7_4702_k82.csv",
    "/data/loto7_4702_k82_loto_2973.csv",
    "/data/loto7_4702_k82_loto_plus_1729.csv",
]

CONTEXT = 20
TRAIN = 384
VALIDATION = 48
TEST = 64
BATCH = 64
CHECKPOINTS = (10, 20, 30)


def read_rows(path):
    rows = []

    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for line, cells in enumerate(csv.reader(handle), 1):
            if not cells or all(not x.strip() for x in cells):
                continue

            try:
                values = tuple(sorted(int(x.strip()) for x in cells))
            except ValueError as exc:
                raise ValueError(
                    f"{path}, red {line}: ocekujem 7 celih brojeva."
                ) from exc

            if len(values) != 7 or len(set(values)) != 7:
                raise ValueError(
                    f"{path}, red {line}: potrebno je 7 razlicitih brojeva."
                )

            if values[0] < 1 or values[-1] > 39:
                raise ValueError(
                    f"{path}, red {line}: dozvoljen opseg je 1..39."
                )

            rows.append(values)

    minimum = CONTEXT + TRAIN + VALIDATION + TEST

    if len(rows) < minimum:
        raise ValueError(
            f"{path}: potrebno najmanje {minimum} kola; ima {len(rows)}."
        )

    return np.asarray(rows, dtype=np.int32)


def examples(rows, start, stop):
    # Za cilj t ulaz sadrzi iskljucivo kola t-20,...,t-1.
    if not (CONTEXT <= start < stop <= len(rows)):
        raise ValueError("Neispravan vremenski interval.")

    x = np.stack([
        rows[t - CONTEXT:t]
        for t in range(start, stop)
    ])
    x = x.astype(np.float32) / np.float32(39.0)

    # Oznaka pripadnosti brojeva stvarnom sledecem kolu.
    # Ovde se ne racunaju istorijske frekvencije.
    y = np.zeros((stop - start, 39), dtype=np.float32)

    for i, t in enumerate(range(start, stop)):
        y[i, rows[t] - 1] = 1.0

    return x, y


def top_seven(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)

    if (
        p.ndim != 2
        or p.shape[1] != 39
        or not np.isfinite(p).all()
    ):
        raise ValueError("Neispravan izlaz modela.")

    # Stabilan izbor: kod jednakog skora prednost ima manji broj.
    order = np.argsort(
        -p, axis=1, kind="stable"
    )[:, :7] + 1

    return np.sort(order, axis=1)


def hits(predictions, actual):
    return np.asarray([
        len(set(map(int, a)) & set(map(int, b)))
        for a, b in zip(predictions, actual)
    ], dtype=np.int32)


def binary_loss(y, p):
    p = np.clip(
        np.asarray(p, dtype=np.float64),
        1e-7,
        1 - 1e-7,
    )

    return float(
        -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    )


def analytic_weights(shape, tag):
    # Analiticka inicijalizacija bez slucajnog generatora.
    # Ovo su pocetne tezine mreze, ne raspodela loto kombinacija.
    a, b = shape

    i = np.arange(1, a + 1, dtype=np.float64)[:, None]
    j = np.arange(1, b + 1, dtype=np.float64)[None, :]

    phase = (
        i * j * np.sqrt(2.0)
        + i * np.sqrt(3.0)
        + j * np.sqrt(5.0)
        + tag
    )

    return (
        np.sqrt(6.0 / (a + b)) * np.sin(phase)
    ).astype(np.float32)


def load_tensorflow():
    try:
        version = importlib.metadata.version("tensorflow")
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(
            "Nedostaje TensorFlow. Koristi navedeno LSTM okruzenje."
        )

    major_minor = tuple(
        int(x) for x in version.split(".")[:2]
    )

    if (
        major_minor < (2, 18)
        and int(np.__version__.split(".")[0]) >= 2
    ):
        raise SystemExit(
            f"TensorFlow {version} i NumPy {np.__version__} "
            "nisu podrzana kombinacija. "
            "Za ovu V1 koristi Python 3.11, "
            "tensorflow==2.16.2 i numpy==1.26.4."
        )

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            f"TensorFlow se ne ucitava: {exc}"
        )

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()

    return tf


def new_model(tf):
    tf.keras.backend.clear_session()
    const = tf.keras.initializers.Constant

    def lstm(inputs, units, tag, sequences):
        bias = np.zeros(4 * units, dtype=np.float32)
        bias[units:2 * units] = 1.0

        return tf.keras.layers.LSTM(
            units,
            return_sequences=sequences,
            kernel_initializer=const(
                analytic_weights((inputs, 4 * units), tag)
            ),
            recurrent_initializer=const(
                analytic_weights((units, 4 * units), tag + 1)
            ),
            bias_initializer=const(bias),
            unit_forget_bias=False,
            dropout=0.0,
            recurrent_dropout=0.0,
            seed=0,
        )

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(CONTEXT, 7)),

        lstm(7, 128, 1.0, True),
        lstm(128, 64, 3.0, False),

        tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_initializer=const(
                analytic_weights((64, 64), 5.0)
            ),
            bias_initializer="zeros",
        ),

        tf.keras.layers.Dense(
            39,
            activation="sigmoid",
            kernel_initializer=const(
                analytic_weights((64, 39), 6.0)
            ),
            bias_initializer="zeros",
        ),
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0,
    )
    loss_function = tf.keras.losses.BinaryCrossentropy()

    @tf.function(input_signature=[
        tf.TensorSpec([None, CONTEXT, 7], tf.float32),
        tf.TensorSpec([None, 39], tf.float32),
    ])
    def step(x, y):
        with tf.GradientTape() as tape:
            loss = loss_function(
                y, model(x, training=True)
            )

        gradients = tape.gradient(
            loss, model.trainable_variables
        )

        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables)
        )

        return loss

    return model, step


def epoch(step, x, y):
    total = 0.0

    # Hronoloski redosled; nema shuffle operacije.
    for start in range(0, len(x), BATCH):
        stop = min(start + BATCH, len(x))

        loss = float(
            step(x[start:stop], y[start:stop]).numpy()
        )

        if not np.isfinite(loss):
            raise RuntimeError(
                "Trening je proizveo neispravnu vrednost gubitka."
            )

        total += loss * (stop - start)

    return total / len(x)


def training_data(rows, end):
    return examples(
        rows,
        max(CONTEXT, end - TRAIN),
        end,
    )


def fit_fixed(tf, rows, end, epochs):
    x, y = training_data(rows, end)
    model, step = new_model(tf)

    for _ in range(epochs):
        epoch(step, x, y)

    return model


def longest_run(numbers):
    best = current = 1

    for a, b in zip(numbers, numbers[1:]):
        current = current + 1 if b == a + 1 else 1
        best = max(best, current)

    return best


def check_data(rows):
    start = len(rows) - TEST - VALIDATION
    x, y = examples(rows, start, start + 2)

    expected = (
        rows[start - CONTEXT:start].astype(np.float32)
        / np.float32(39)
    )

    assert np.array_equal(x[0], expected)
    assert np.array_equal(
        np.flatnonzero(y[0]) + 1,
        rows[start],
    )

    # Promena ciljnog i buducih kola ne sme menjati ulaz.
    altered = rows.copy()
    altered[start:] = np.arange(1, 8)

    other, _ = examples(altered, start, start + 1)
    assert np.array_equal(x[:1], other)

    assert tuple(
        top_seven(np.zeros((1, 39)))[0]
    ) == tuple(range(1, 8))

    assert np.array_equal(
        analytic_weights((7, 512), 1),
        analytic_weights((7, 512), 1),
    )


def run(tf, path, rows):
    n = len(rows)
    test_start = n - TEST
    validation_start = test_start - VALIDATION

    print(
        f"\n{'=' * 64}\n"
        f"{Path(path).name} | kola: {n}",
        flush=True,
    )

    x, y = training_data(rows, validation_start)
    vx, vy = examples(
        rows, validation_start, test_start
    )

    model, step = new_model(tf)
    candidates = []

    for number in range(1, max(CHECKPOINTS) + 1):
        epoch(step, x, y)

        if number in CHECKPOINTS:
            probabilities = model(
                vx, training=False
            ).numpy()

            mean_hits = float(
                hits(
                    top_seven(probabilities),
                    rows[validation_start:test_start],
                ).mean()
            )

            loss = binary_loss(vy, probabilities)

            # Prvo pogodak; zatim BCE; zatim manje epoha.
            candidates.append(
                (mean_hits, -loss, -number)
            )

            print(
                f"Validacija: epoha {number:2d} | "
                f"pogodaka {mean_hits:.4f} | "
                f"BCE {loss:.6f}",
                flush=True,
            )

    selected = -max(candidates)[2]

    del step, model
    gc.collect()

    # Broj epoha bira se pre testa.
    # Tezine su zamrznute tokom svih 64 test-kola.
    # Ulazi svakog test-kola sadrze samo tada poznata kola.
    model = fit_fixed(
        tf, rows, test_start, selected
    )

    tx, _ = examples(rows, test_start, n)

    predictions = top_seven(
        model(tx, training=False).numpy()
    )

    assert np.array_equal(
        predictions,
        top_seven(model(tx, training=False).numpy()),
    )

    actual = rows[test_start:n]
    values = hits(predictions, actual)

    control = hits(
        rows[test_start - 1:n - 1],
        actual,
    )

    replay = 0

    for offset, prediction in enumerate(predictions):
        t = test_start + offset

        replay += bool(
            np.any(
                np.all(rows[:t] == prediction, axis=1)
            )
        )

    print(f"Izabrano epoha: {selected}")

    print(
        f"TEST {TEST}: srednji broj pogodaka = "
        f"{values.mean():.4f}"
    )

    print(
        "Kontrola: prethodno izvuceno kolo = "
        f"{control.mean():.4f}"
    )

    print(
        "Raspodela pogodaka 0..7:",
        np.bincount(values, minlength=8).tolist(),
    )

    print(
        f"3+: {int((values >= 3).sum())}/{TEST}; "
        f"4+: {int((values >= 4).sum())}/{TEST}"
    )

    print(
        "Razlicitih predloga: "
        f"{len(set(map(tuple, predictions)))}/{TEST}"
    )

    print(
        "Predloga prisutnih u prethodnoj istoriji: "
        f"{replay}/{TEST}"
    )

    print(
        "Predloga sa nizom 5+:",
        sum(
            longest_run(p) >= 5
            for p in predictions
        ),
    )

    print(
        "Predloga potpuno u jednoj polovini:",
        sum(
            bool(p[-1] <= 19 or p[0] >= 20)
            for p in predictions
        ),
    )

    del model
    gc.collect()

    # Za NEXT se isti model uci iznova na poslednjem
    # dostupnom prozoru, sa vec izabranim brojem epoha.
    model = fit_fixed(tf, rows, n, selected)

    nx = (
        rows[-CONTEXT:].astype(np.float32)[None, :, :]
        / np.float32(39)
    )

    probabilities = model(
        nx, training=False
    ).numpy()

    prediction = top_seven(probabilities)[0]

    assert np.array_equal(
        prediction,
        top_seven(
            model(nx, training=False).numpy()
        )[0],
    )

    historical = bool(
        np.any(
            np.all(rows == prediction, axis=1)
        )
    )

    print(
        f"NEXT, red {n + 1}: "
        + ",".join(map(str, prediction))
    )

    print(
        "Postoji u istoriji: "
        + ("DA" if historical else "NE")
    )

    print(
        f"Najduzi uzastopni niz: {longest_run(prediction)}; "
        f"brojeva 1..19: {int((prediction <= 19).sum())}"
    )

    if values.mean() <= control.mean():
        print(
            "Na ovom testu model nije nadmasio kontrolu."
        )

    del model

    return tuple(map(int, prediction))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LotteryForecast LSTM V1 — "
            "tri nezavisna CSV-a"
        )
    )

    parser.add_argument(
        "csv",
        nargs="*",
        help="CSV fajlovi, bez zaglavlja",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Provera podataka bez TensorFlow treninga",
    )

    args = parser.parse_args()

    datasets = [
        (path, read_rows(path))
        for path in (args.csv or DEFAULTS)
    ]

    for path, rows in datasets:
        check_data(rows)

        print(
            f"Provera OK: {Path(path).name} | "
            f"{len(rows)} kola",
            flush=True,
        )

    if args.check_only:
        return

    tf = load_tensorflow()

    print(
        "LSTM 128 -> LSTM 64 -> Dense 64 -> "
        "39 sigmoid izlaza."
    )

    print(
        "Bez random inicijalizacije, mesanja uzoraka "
        "i generisanja kombinacija."
    )

    results = [
        (path, run(tf, path, rows))
        for path, rows in datasets
    ]

    # Isti rezultat se prijavljuje, ne menja se vestacki.
    for i, (path, combo) in enumerate(results):
        for other_path, other_combo in results[:i]:
            if combo == other_combo:
                print(
                    f"Isti NEXT: {Path(path).name} i "
                    f"{Path(other_path).name}: {combo}"
                )


if __name__ == "__main__":
    main()



"""
Provera OK: loto7_4702_k82.csv | 4702 kola
Provera OK: loto7_4702_k82_loto_2973.csv | 2973 kola
Provera OK: loto7_4702_k82_loto_plus_1729.csv | 1729 kola
LSTM 128 -> LSTM 64 -> Dense 64 -> 39 sigmoid izlaza.
Bez random inicijalizacije, mesanja uzoraka i generisanja kombinacija.


================================================================
loto7_4702_k82.csv | kola: 4702
Validacija: epoha 10 | pogodaka 1.2083 | BCE 0.471858
Validacija: epoha 20 | pogodaka 1.1458 | BCE 0.471600
Validacija: epoha 30 | pogodaka 1.1875 | BCE 0.471582
Izabrano epoha: 10
TEST 64: srednji broj pogodaka = 1.2656
Kontrola: prethodno izvuceno kolo = 1.1719
Raspodela pogodaka 0..7: [13, 24, 24, 3, 0, 0, 0, 0]
3+: 3/64; 4+: 0/64
Razlicitih predloga: 1/64
Predloga prisutnih u prethodnoj istoriji: 0/64
Predloga sa nizom 5+: 0
Predloga potpuno u jednoj polovini: 0
NEXT, red 4703: 5,x,8,y,18,z,34
Postoji u istoriji: NE
Najduzi uzastopni niz: 2; brojeva 1..19: 5



================================================================
loto7_4702_k82_loto_2973.csv | kola: 2973
Validacija: epoha 10 | pogodaka 1.3333 | BCE 0.472332
Validacija: epoha 20 | pogodaka 1.1667 | BCE 0.472272
Validacija: epoha 30 | pogodaka 1.1458 | BCE 0.472271
Izabrano epoha: 10
TEST 64: srednji broj pogodaka = 1.2188
Kontrola: prethodno izvuceno kolo = 1.3281
Raspodela pogodaka 0..7: [19, 20, 18, 6, 1, 0, 0, 0]
3+: 7/64; 4+: 1/64
Razlicitih predloga: 1/64
Predloga prisutnih u prethodnoj istoriji: 0/64
Predloga sa nizom 5+: 0
Predloga potpuno u jednoj polovini: 0
NEXT, red 2974: 7,x,16,y,26,z,38
Postoji u istoriji: NE
Najduzi uzastopni niz: 2; brojeva 1..19: 4
Na ovom testu model nije nadmasio kontrolu.



================================================================
loto7_4702_k82_loto_plus_1729.csv | kola: 1729
Validacija: epoha 10 | pogodaka 1.3125 | BCE 0.473638
Validacija: epoha 20 | pogodaka 1.2292 | BCE 0.473146
Validacija: epoha 30 | pogodaka 1.2292 | BCE 0.473173
Izabrano epoha: 10
TEST 64: srednji broj pogodaka = 1.1875
Kontrola: prethodno izvuceno kolo = 1.2656
Raspodela pogodaka 0..7: [12, 32, 17, 2, 1, 0, 0, 0]
3+: 3/64; 4+: 1/64
Razlicitih predloga: 1/64
Predloga prisutnih u prethodnoj istoriji: 0/64
Predloga sa nizom 5+: 0
Predloga potpuno u jednoj polovini: 0
NEXT, red 1730: 3,x,10,y,24,z,34
Postoji u istoriji: NE
Najduzi uzastopni niz: 2; brojeva 1..19: 3
Na ovom testu model nije nadmasio kontrolu.
"""



"""
- LSTM 128 → LSTM 64 
"""
