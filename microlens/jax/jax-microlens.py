import os
import sys

import imageio.v3 as iio
# import skvideo.io

video_dir = "/blok/@data/cn/edu/westlake/recsys/MicroLens-100k-Dataset/MicroLens-100k_videos"

if __name__=="__main__":
    for filename in os.listdir(video_dir):
        print(filename)
        sys.stdout.flush()
        if filename.endswith('.mp4'):
            path = os.path.join(video_dir, filename)

            print(path)
            sys.stdout.flush()

            # videodata = skvideo.io.vread(path)
            # print(videodata.shape)

            # # read a single frame
            # frame = iio.imread(
            #     path,
            #     index=42,
            #     plugin="pyav",
            # )

            # bulk read all frames
            # Warning: large videos will consume a lot of memory (RAM)
            # frames = iio.imread(path, plugin="pyav")

            # iterate over large videos
            for frame in iio.imiter(path):#, plugin="pyav"):
                print(frame.shape, frame.dtype)
