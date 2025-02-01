import os
import shutil
import random

def split_dataset(train_dir='dataset/train', test_dir='dataset/test', test_ratio=0.2):
    """
    Split the dataset into train and test sets
    """
    # Create test directory if it doesn't exist
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Get all categories
    categories = os.listdir(train_dir)
    
    for category in categories:
        # Create category directory in test
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)
        
        # Get all images in category
        category_path = os.path.join(train_dir, category)
        images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Calculate number of test images
        num_test = max(1, int(len(images) * test_ratio))
        
        # Randomly select test images
        test_images = random.sample(images, num_test)
        
        # Move test images to test directory
        for image in test_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(test_dir, category, image)
            shutil.copy2(src, dst)
        
        print(f"{category}: {num_test} images moved to test set")

if __name__ == "__main__":
    print("Splitting dataset into train and test sets...")
    split_dataset()
    print("Dataset split complete!") 