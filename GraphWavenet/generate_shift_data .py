import numpy as np
import pandas as pd
import argparse

def generate_shift_data(args):
    df = pd.read_hdf(args.traffic_df_filename)

    # Build features: speed + time-of-day, same as standard generate_training_data.py
    time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day = np.tile(time_ind, [1, df.shape[1], 1]).transpose((2, 1, 0))
    feature_list = [np.expand_dims(df.values, axis=-1), time_in_day]
    data = np.concatenate(feature_list, axis=-1).astype(np.float32)

    x_offsets = np.sort(np.arange(-11, 1, 1))
    y_offsets = np.sort(np.arange(1, 13, 1))

    x, y, sample_days = [], [], []
    min_t = abs(min(x_offsets))
    max_t = data.shape[0] - abs(max(y_offsets))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
        sample_days.append(df.index[t].dayofweek)  # 0=Monday ... 6=Sunday

    x = np.stack(x, axis=0).astype(np.float32)
    y = np.stack(y, axis=0).astype(np.float32)
    sample_days = np.array(sample_days)

    is_weekday = sample_days < 5  # Mon-Fri
    is_weekend = ~is_weekday

    x_weekday, y_weekday = x[is_weekday], y[is_weekday]
    x_test, y_test = x[is_weekend], y[is_weekend]  # weekend = test set

    # 10% random holdout from weekday data for validation
    n_weekday = x_weekday.shape[0]
    rng = np.random.default_rng(42)
    val_idx = rng.choice(n_weekday, size=int(0.1 * n_weekday), replace=False)
    train_idx = np.setdiff1d(np.arange(n_weekday), val_idx)

    x_train, y_train = x_weekday[train_idx], y_weekday[train_idx]
    x_val, y_val = x_weekday[val_idx], y_weekday[val_idx]

    print("train x:", x_train.shape, "y:", y_train.shape)
    print("val x:", x_val.shape, "y:", y_val.shape)
    print("test (weekend) x:", x_test.shape, "y:", y_test.shape)

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    for cat in ["train", "val", "test"]:
        _x, _y = locals()[f"x_{cat}"], locals()[f"y_{cat}"]
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x, y=_y,
            x_offsets=x_offsets.reshape(-1, 1),
            y_offsets=y_offsets.reshape(-1, 1),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--traffic_df_filename", type=str, required=True)
    args = parser.parse_args()
    generate_shift_data(args)
