import sklearn.datasets
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

if __name__ == "__main__":

    data_list = [
        {
            16: 26033,
            32: 55065,
            48: 42437,
            64: 37160,
            80: 19379,
            96: 26521,
            112: 14779,
            128: 12400,
            144: 7744,
            160: 5859,
            176: 3937,
            192: 2963,
            208: 2015,
            224: 1562,
            240: 1409,
            256: 2881,
        },
        {
            16: 33614,
            32: 69297,
            48: 48671,
            64: 38494,
            80: 16566,
            96: 23687,
            112: 10228,
            128: 8405,
            144: 4223,
            160: 3032,
            176: 1917,
            192: 1310,
            208: 779,
            224: 509,
            240: 448,
            256: 964,
        },
        {
            16: 36809,
            32: 74420,
            48: 50271,
            64: 38313,
            80: 15351,
            96: 21834,
            112: 8614,
            128: 6962,
            144: 3208,
            160: 2411,
            176: 1357,
            192: 873,
            208: 539,
            224: 334,
            240: 276,
            256: 572,
        },
        {
            16: 38526,
            32: 76977,
            48: 50766,
            64: 37806,
            80: 14734,
            96: 20832,
            112: 7899,
            128: 6424,
            144: 2947,
            160: 2046,
            176: 1136,
            192: 728,
            208: 416,
            224: 249,
            240: 231,
            256: 427,
        },
        {
            16: 39585,
            32: 78628,
            48: 50929,
            64: 37535,
            80: 14358,
            96: 20153,
            112: 7359,
            128: 6173,
            144: 2689,
            160: 1941,
            176: 981,
            192: 628,
            208: 385,
            224: 222,
            240: 215,
            256: 363,
        },
        {
            16: 41689,
            32: 81234,
            48: 51211,
            64: 36819,
            80: 13684,
            96: 18987,
            112: 6811,
            128: 5425,
            144: 2359,
            160: 1675,
            176: 836,
            192: 532,
            208: 292,
            224: 185,
            240: 152,
            256: 253,
        }
    ]

    # Determine all possible x-axis keys across all dictionaries
    all_keys = sorted({key for d in data_list for key in d.keys()})

    # Determine global y-axis max for uniform scaling
    global_max = max(max(d.values()) for d in data_list)

    # Create subplots
    fig, axes = plt.subplots(len(data_list), 1, figsize=(8, 1.2 * len(data_list)), sharex=True)

    if len(data_list) == 1:
        axes = [axes]

    for ax, data in zip(axes, data_list):
        # Get values in the order of all_keys (fill missing keys with 0)
        values = [data.get(k, 0) for k in all_keys]
        ax.bar(all_keys, values, width=10, color='skyblue', edgecolor='black')
        ax.set_ylim(0, global_max)
        ax.set_ylabel('Count')

    # Label each x-axis tick with the key
    axes[-1].set_xticks(all_keys)
    axes[-1].set_xlabel('Bin')

    plt.tight_layout()
    # plt.show()
    plt.savefig('foo.png', bbox_inches='tight', dpi=600)
