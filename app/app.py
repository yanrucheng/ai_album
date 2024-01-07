import argparse
from image_similarity import ImageSimilarity
from utils import copy_file_as_cluster
import pprint

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Similarity Script')

    # Add arguments
    parser.add_argument('folder_path', type=str, help='Path to the folder containing images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing images (default: 16)')
    parser.add_argument('--show_progress', action='store_true', help='Show progress bar during processing')
    parser.add_argument('--distance_levels', nargs='*', type=float, default=[2, 0.5], help='List of distance levels for hierarchical clustering (default: [2, 0.5])')
    parser.add_argument('--output_path', type=str, default='', help='Output path to copy files as clusters (default: ./data/testoutput/)')

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Usage
    s = ImageSimilarity(args.folder_path,
                        batch_size=args.batch_size,
                        show_progress_bar=args.show_progress)

    # Compute all captions
    s.compute_all_captions()

    # Compute all tags
    s.compute_all_tags()

    # Clustering images with specified distance levels
    clusters = s.cluster_images_with_multilevel_hierarchical(distance_levels=args.distance_levels)

    if args.output_path:
        # Copying files as clusters to the specified output path
        copy_file_as_cluster(clusters, args.output_path)

    # Additional logic can be added here

if __name__ == "__main__":
    main()
