import numpy
import torch
import os
import coopscenes as cs

from models import StixelPredictor


def main():
    DATAPATH = "data/seq_1"
    WEIGHTS_PATH = "models/StixelNExT-Pro_fresh-durian-552_9.pth"

    # Model
    stixel_predictor = StixelPredictor(weights=WEIGHTS_PATH)

    # Dataset
    dataset = cs.Dataloader("/data/seq_1")
    """
    frames = []
    for datarecord in dataset:
        for frame in datarecord:
            frames.append(frame)
    """
    example_record = cs.DataRecord(os.path.join(DATAPATH, "id01742_2024-09-27_10-39-26.4mse"))
    example_frame = example_record[27]
    print(example_frame.frame_id)
    example_frame.vehicle.cameras.STEREO_LEFT.show()
    example_frame.tower.cameras.VIEW_1.show()


if __name__ == "__main__":
    main()
