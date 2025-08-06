import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

def show_class_distribution(df):
    plt.figure(figsize=(20, 6))
    sns.countplot(data=df, x="class_name", order=df["class_name"].value_counts().index)
    plt.xticks(rotation=90)
    plt.title("Image Count per Class")
    plt.tight_layout()
    plt.show()

def show_sample_images(df, class_name, n=5):
    samples = df[df['class_name'] == class_name].sample(n)
    plt.figure(figsize=(15, 3))
    for i, row in enumerate(samples.itertuples()):
        img = mpimg.imread(row.path)
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(class_name)
    plt.show()
    
def plot_history(histories, titles=['Initial Training', 'Fine-tuning']):
    for history, title in zip(histories, titles):
        plt.plot(history.history['val_accuracy'], label=f'{title} Val Acc')
        plt.plot(history.history['accuracy'], label=f'{title} Train Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.title("Model Training Progress")
    plt.show()
