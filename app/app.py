from media_center import MediaCenter
from my_cluster import copy_file_as_cluster
import pprint
import utils
from function_tracker import global_tracker
from argparser import parse_arguments, to_default_output_path
import logging


def app(in_folder, args):
    # Usage
    s = MediaCenter(in_folder,
                    batch_size = args.batch_size,
                    check_rotation = not args.disable_rotation,
                    check_nude = not args.disable_explicity_detection,
                    show_progress_bar = not args.disable_progress,
                    cache_flags = args.cache_flags,
                    language = args.language,
                    )

    # Compute all captions
    s.compute_all_captions()

    # Compute all tags
    s.compute_all_tags()

    # Clustering images with specified distance levels
    clusters = s.cluster(*args.distance_levels)

    output_path = args.output_path
    if output_path == '':
        output_path = to_default_output_path(in_folder)

    if 'print' in args.output_type:
        pprint.pprint(clusters)

    if 'original' in args.output_type:
        copy_file_as_cluster(clusters, output_path,
                             operator = s.copy_with_meta_rotate)

    if 'thumbnail' in args.output_type:
        thumb_clusters = s.cluster_to_thumbnail(clusters)
        copy_file_as_cluster(thumb_clusters, output_path)

    if 'link' in args.output_type:
        copy_file_as_cluster(clusters, output_path,
                             operator = utils.create_relative_symlink)


def main():
    args = parse_arguments()

    if args.debug:
        global_tracker.enable()
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    for f in args.folder_paths:
        app(f, args)

    if args.debug:
        global_tracker.report()


if __name__ == "__main__":
    main()
