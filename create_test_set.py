import os
import shutil
import random

def create_test_set(categories_to_test=10, images_per_category=5):
    # Base directories
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'
    
    # Create test directory if it doesn't exist
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Get all categories
    categories = sorted(os.listdir(train_dir))
    
    # Select random categories if we have more than we want to test
    if len(categories) > categories_to_test:
        categories = random.sample(categories, categories_to_test)
    
    print(f"\nCreating test set with {len(categories)} categories:")
    
    for category in categories:
        # Create category directory in test
        test_category_path = os.path.join(test_dir, category)
        if not os.path.exists(test_category_path):
            os.makedirs(test_category_path)
        
        # Get all images from training category
        train_category_path = os.path.join(train_dir, category)
        images = [f for f in os.listdir(train_category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Select random images
        selected_images = random.sample(images, min(len(images), images_per_category))
        
        # Copy images to test directory
        for image in selected_images:
            src = os.path.join(train_category_path, image)
            dst = os.path.join(test_category_path, image)
            shutil.copy2(src, dst)
        
        print(f"- {category}: copied {len(selected_images)} images")

if __name__ == "__main__":
    print("Creating test dataset...")
    create_test_set(categories_to_test=10, images_per_category=5)
    print("\nTest dataset creation complete!") 