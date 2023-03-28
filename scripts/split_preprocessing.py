import os
import shutil
import random
# Set the seed for reproducibility
random.seed(42)

# Split the NCaltech raw data into train, validation and test sets- 80% train, 10% val, 10% test

# This is the original dir structure:
# Caltech101/
    # accordion/
    #     image_0001.bin
    #     ...
    # anchor/
    #     image_0001.bin
    #     ...
    # ...
# From this, split by copying the data into train, validation and test sets (80%, 10%, 10%):
# ncaltech101/
#     training/
#         accordion/
#             image_0001.bin
#             ...
#         anchor/
#         ... 
#     validation/
#         ...
#     test/
#         ...


original_dir = 'Caltech101'
# Cleanup .DS_Store files recursively in all directories
# for root, dirs, files in os.walk(original_dir):
#     for file in files:
#         if file == '.DS_Store':
#             os.remove(os.path.join(root, file))

# Make the new directory, exit if it already exists
new_dir = 'ncaltech101'
if os.path.exists(new_dir):
    print('The directory `ncaltech101` already exists. Please delete it and try again.')
    exit()
os.mkdir(new_dir)


# Make the train, validation and test directories
train_dir = os.path.join(new_dir, 'training')
val_dir = os.path.join(new_dir, 'validation')
test_dir = os.path.join(new_dir, 'test')

os.mkdir(train_dir)
os.mkdir(val_dir)
os.mkdir(test_dir)


# Make a directory for each class in the train, validation and test directories
# At the same time, get the number of images in each class
class_counts = {}
for class_dir in os.listdir(original_dir):
    os.mkdir(os.path.join(train_dir, class_dir))
    os.mkdir(os.path.join(val_dir, class_dir))
    os.mkdir(os.path.join(test_dir, class_dir))

    class_counts[class_dir] = len(os.listdir(os.path.join(original_dir, class_dir)))


# Copy the data into the train, validation and test directories (80%, 10%, 10%)
for class_dir in os.listdir(original_dir):
    images = os.listdir(os.path.join(original_dir, class_dir))
    random.shuffle(images)

    train_images = images[:int(0.8 * class_counts[class_dir])]
    val_images = images[int(0.8 * class_counts[class_dir]):int(0.9 * class_counts[class_dir])]
    test_images = images[int(0.9 * class_counts[class_dir]):]

    for image in train_images:
        shutil.copyfile(os.path.join(original_dir, class_dir, image), os.path.join(train_dir, class_dir, image))

    for image in val_images:
        shutil.copyfile(os.path.join(original_dir, class_dir, image), os.path.join(val_dir, class_dir, image))

    for image in test_images:
        shutil.copyfile(os.path.join(original_dir, class_dir, image), os.path.join(test_dir, class_dir, image))


# Check that the data has been split correctly
for split in [train_dir, val_dir, test_dir]:
    total = 0
    for class_dir in os.listdir(split):
        total += len(os.listdir(os.path.join(split, class_dir)))
    print(split, total)
