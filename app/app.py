import argparse
from media_similarity import MediaSimilarity
from my_cluster import copy_file_as_cluster
import pprint
import os
import utils

def parse_arguments():
    parser = argparse.ArgumentParser(description='AI-Album, an LLM-based AI auto media grouper')

    # Add arguments
    parser.add_argument('folder_path', type=str,
                        help='Path to the folder containing images')
    parser.add_argument('-b', '--batch-size', type=int, default=16,
                        help='Batch size for processing images (default: 16)')
    parser.add_argument('-qon', '--questionare-on', type=bool, default=False,
                        help='Turn on calculation for image questionare. This adds 8G additional memory (default: False)')
    parser.add_argument('-sp', '--show-progress', action='store_true',
                        help='Show progress bar during processing')
    parser.add_argument('-dl', '--distance-levels', nargs='*',
                        type=float, default=[2, 0.5],
                        help='List of distance levels for hierarchical clustering (default: [2, 0.5])')
    parser.add_argument('-o', '--output-path', type=validate_output_folder, default='',
                        help="Output path to copy files as clusters (default: ''). Could be path, print, default (<input_dir>_clustered), or ''")
    parser.add_argument('-ot', '--output-type', nargs='+',
                        choices=['thumbnail', 'original', 'link'],
                        default=['thumbnail', 'link'],
                        help='Output types can be (one/multiple of)thumbnail, original, or link')

    args = parser.parse_args()

    if args.output_path == 'default':
        args.output_path = to_default_output_path(args.folder_path)

    return args

def to_default_output_path(in_path):
    return in_path.rstrip('/').rstrip('\\') + '_clustered'

def validate_output_folder(path):
    if path in ('', 'print', 'default'): return path
    if os.path.isdir(path): return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid path or 'print' or empty")


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

    if not args.output_path:
        return

    # Clustering images with specified distance levels
    clusters = s.cluster(*args.distance_levels)

    if args.output_path == 'print':
        pprint.pprint(clusters)
        return

    if 'original' in args.output_type:
        copy_file_as_cluster(clusters, args.output_path)
    if 'thumbnail' in args.output_type:
        thumb_clusters = s.cluster_to_thumbnail(clusters)
        copy_file_as_cluster(thumb_clusters, args.output_path)
    if 'link' in args.output_type:
        copy_file_as_cluster(clusters,
                             args.output_path,
                             operator = utils.create_relative_symlink,
                             )


if __name__ == "__main__":
    main()
