import os

def print_categories():
    train_path = 'dataset/train'
    if not os.path.exists(train_path):
        print(f"Error: {train_path} directory not found!")
        return
    
    categories = sorted(os.listdir(train_path))
    print("\nCurrent training categories:")
    for cat in categories:
        print(f"- {cat}")
        # Also print number of images in each category
        category_path = os.path.join(train_path, cat)
        if os.path.isdir(category_path):
            num_images = len([f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  ({num_images} images)")

if __name__ == "__main__":
    print_categories() 