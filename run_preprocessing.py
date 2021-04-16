import sys

sys.path.append('./preprocessing')
from pipeline_mst import PipelineMST
from remove_id import RemoveId

# Setup path helpers and file paths
get_raw_path = lambda file : f"./data/raw/{file}"
get_prep_path = lambda file: f"./data/preprocessed/{file}"

# TODO: Convert path data structure to dict
input_full_train_pos_path = get_raw_path("train_pos_full.txt")
input_full_train_neg_path = get_raw_path("train_neg_full.txt")
input_part_train_pos_path = get_raw_path("part_train_pos.txt")
input_part_train_neg_path = get_raw_path("part_train_neg.txt")
input_test_path = get_raw_path("test_data.txt")

output_full_train_pos_path = get_prep_path("train_pos_full.txt")
output_full_train_neg_path = get_prep_path("train_neg_full.txt")
output_part_train_pos_path = get_prep_path("part_train_pos.txt")
output_part_train_neg_path = get_prep_path("part_train_neg.txt")
output_test_path_noid = get_prep_path("test_prep_noid.txt")
output_test_path = get_prep_path("test_prep.txt")

def run_preprocessing(is_train=False, is_full=True):
  preprocessing = PipelineMMST()

  if is_train and is_full:
    preprocessing.process(
      [ input_full_train_pos_path, input_full_train_neg_path ],
      [ output_full_train_pos_path, output_full_train_neg_path ]
    )
  elif is_train and not is_full:
    preprocessing.process(
      [ input_part_train_pos_path, input_part_train_neg_path ],
      [ output_part_train_pos_path, output_part_train_neg_path ]
    )
  else:
    ri = RemoveId()
    ri.set_paths(input_test_path, output_test_path_noid)
    ri.run()

    preprocessing.process(
      [ output_test_path_noid ],
      [ output_test_path ]
    )

# Preprocess
if __name__ == '__main__':
  run_preprocessing(True, False)
