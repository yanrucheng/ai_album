import argparse
from media_similarity import MediaSimilarity
from my_cluster import copy_file_as_cluster
import pprint

def parse_arguments():
    parser = argparse.ArgumentParser(description='AI-Album, an LLM-based AI auto media grouper')

    # Add arguments
    parser.add_argument('folder_path', type=str,
                        help='Path to the folder containing images')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Batch size for processing images (default: 16)')
    parser.add_argument('-qon', '--questionare_on', type=bool, default=False,
                        help='Turn on calculation for image questionare. This adds 8G additional memory (default: False)')
    parser.add_argument('-sp', '--show_progress', action='store_true',
                        help='Show progress bar during processing')
    parser.add_argument('-dl', '--distance_levels', nargs='*',
                        type=float, default=[2, 0.5],
                        help='List of distance levels for hierarchical clustering (default: [2, 0.5])')
    parser.add_argument('-o', '--output_path', type=str, default='',
                        help='Output path to copy files as clusters (default: ./data/testoutput/)')
    parser.add_argument('-ot', '--output_type', choices=['thumbnail', 'original'],
                        default='thumbnail',
                        help='The type of output: "thumbnail" or "original"')

    args = parser.parse_args()
    if args.output_path == '':
        args.output_type = 'original'
    return args

def main():
    args = parse_arguments()

    # Usage
    s = MediaSimilarity(args.folder_path,
                        batch_size=args.batch_size,
                        questionare_on=args.questionare_on,
                        show_progress_bar=args.show_progress)

    # Compute all captions
    s.compute_all_captions()

    # Compute all tags
    s.compute_all_tags()

    # Clustering images with specified distance levels
    clusters = s.cluster(*args.distance_levels, output_type=args.output_type)

    if args.output_path:
        # Copying files as clusters to the specified output path
        copy_file_as_cluster(clusters, args.output_path)
    else:
        pprint.pprint(clusters)

    # Additional logic can be added here

if __name__ == "__main__":
    main()
