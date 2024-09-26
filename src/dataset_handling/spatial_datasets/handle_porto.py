import polars as pl
import numpy as np
from tqdm import tqdm

SAMPLE_S8_F16_PATH = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/spatial_datasets/porto_s8_f16.bin"
SAMPLE_S8_F32_PATH = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/spatial_datasets/porto_s8_f32.bin"
SAMPLE_S80_F32_PATH = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/spatial_datasets/porto_s80_f32.bin"
SAMPLE_S640_F32_PATH = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/spatial_datasets/porto_s640_f32.bin"
CSV_PATH = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/spatial_datasets/porto.csv"
CSV_S8_SAMPLE_PATH = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/spatial_datasets/porto_s8.csv"
S8_SAMPLE_SIZE = 10427415
S80_SAMPLE_SIZE = 1042742
S640_SAMPLE_SIZE = 130343

def str_to_float(x: str):
    try:
        return float(x)
    except ValueError:
        return None

def process_polyline(polyline, sample_rate):
    results = []
    polyline_split = polyline.strip('[]').split('],[')
    all_coords = [pair.strip("[]'").split(',') for pair in polyline_split]
    for coord_pair in all_coords:
        coord_pair = [str_to_float(coord) for coord in coord_pair]
        if None in coord_pair:
            continue
        if np.random.randint(sample_rate) != 0:
            continue
        results.append((np.float32(coord_pair[0]), np.float32(coord_pair[1])))
    return results


# @njit(parallel=True)
def read_porto_data(sample_rate=8, csv_path=CSV_PATH):
    save_path = f"/home/hphi344/Documents/GS-DBSCAN-Analysis/data/spatial_datasets/porto_s{sample_rate}.csv"
    df = pl.read_csv(csv_path)

    polylines = df['POLYLINE'].to_list()
    results = []

    with tqdm(total=len(polylines), desc="Reading Porto Data") as pbar:
        for polyline in polylines:
            result = process_polyline(polyline, sample_rate)
            results.extend(result)
            pbar.update(1)

    result_df = pl.DataFrame({
        "x": [r[0] for r in results],
        "y": [r[1] for r in results]
    })

    result_df.write_csv(save_path)
    return result_df

#
# def read_porto_data(sample_rate=8):
#     save_path = f"/home/hphi344/Documents/GS-DBSCAN-Analysis/data/spatial_datasets/porto_s{sample_rate}.csv"
#
#     df = pl.read_csv("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/spatial_datasets/porto.csv")
#
#     result_df = pl.DataFrame({
#         "x": pl.Series("x", dtype=pl.Float32),
#         "y": pl.Series("x", dtype=pl.Float32)
#     })
#
#     for polyline in tqdm(df['POLYLINE'], desc="Reading Porto Data"):
#         polyline_split = polyline.strip('[]').split('],[')
#
#         all_coords = [pair.strip("[]'").split(',') for pair in polyline_split]
#
#         for coord_pair in all_coords:
#             coord_pair = [str_to_float(coord) for coord in coord_pair]
#
#             if None in coord_pair:
#                 continue
#
#             if np.random.randint(sample_rate) != 0:
#                 continue
#
#             result_df = result_df.vstack(pl.DataFrame({
#                 "x": [np.float32(coord_pair[0])],
#                 "y": [np.float32(coord_pair[1])]
#             }))
#
#     result_df.write_csv(save_path)
#
#     return result_df

def sample_to_array(sampled_df, save_path, dtype=np.float32):
    array = sampled_df.to_numpy()
    array = array.astype(dtype)

    array.tofile(save_path)

    return sampled_df

def sub_sample_porto(sub_sample_path, sub_sample_rate=80):
    data = np.fromfile(SAMPLE_S8_F32_PATH, dtype=np.float32)

    data = data.reshape(S8_SAMPLE_SIZE, 2)

    np.random.shuffle(data)
    data_sub_sample = data[::sub_sample_rate]

    data_sub_sample.tofile(sub_sample_path)

    print(f"New sample size is {data_sub_sample.shape[0]}")

    return data_sub_sample

if __name__ == "__main__":
    # df = read_porto_data(80, CSV_PATH)

    # sampled_df = pl.read_csv(CSV_PATH)
    #
    # # sample_to_array(sampled_df, SAMPLE_F32_PATH)
    # # sample_to_array(sampled_df, SAMPLE_F16_PATH, dtype=np.float16)
    #
    # print(sampled_df.head())

    data = np.fromfile(SAMPLE_S8_F32_PATH, dtype=np.float32)

    data = data.reshape(S8_SAMPLE_SIZE, 2)

    np.random.shuffle(data)
    data_s640 = data[::80]

    data_s640.tofile(SAMPLE_S640_F32_PATH)

    print(data_s640.shape[0])

