import os
import json
import argparse

#? Parsing command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("--inputdirectory", default=None, type=str, required=True,
                    help="The input directory containing inception tsv parser output JSON files to merge.")
parser.add_argument("--outputfile", default=None, type=str, required=True,
                    help="The output file to write merged data to.")

args = parser.parse_args()
inputdir = args.inputdirectory
outputfile = args.outputfile

#? Merging all examples from all files
merged_examples = []

for file in os.listdir(inputdir):
    if file.endswith(".json"):
        with open(os.path.join(inputdir, file), "r") as in_file:
            example_list = json.load(in_file)
            print('Adding %d examples to training data from file %s' % (len(example_list), file))
            merged_examples.extend(example_list)

print('Number of merged examples: %d' % len(merged_examples))

#? Writing merged examples to output file
with open(outputfile, "w") as out_file:
    json.dump(merged_examples, out_file)