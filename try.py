from utils.data_utils import load_images

def test_loading():
    print("Testing image loading...")
    try:
        originals, counterfeits = load_images()
        print("\nSuccess! Sample image shapes:")
        print(f"Original: {originals[0].shape if originals else 'N/A'}")
        print(f"Counterfeit: {counterfeits[0].shape if counterfeits else 'N/A'}")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    test_loading()