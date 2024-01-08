from image_similarity import ImageSimilarity

def main():
    # Usage
    s = ImageSimilarity('./samples/samples',
                        batch_size=2,
                        show_progress_bar=True)
    print(s.media_fps)

    # Compute all captions
    s.compute_all_captions()

    # Compute all tags
    s.compute_all_tags()

    # Clustering images with specified distance levels
    clusters = s.cluster(distance_levels=[2,0.5])

if __name__ == "__main__":
    main()
