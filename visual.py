import numpy as np
from sklearn.manifold import TSNE 
import pandas as 
import matplotlib.pyplot as plt
import seaborn as sns

def reduce_dimensions(data, column_name):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # reduce using t-SNE
    rows = []
    for i, row in data.iterrows():
      rows.append(np.array(row[column_name]))

    vectors = np.array(rows)
    labels = np.array(data['label'])
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    
    return x_vals, y_vals, 
    
def to_nums(x):
    return [float(i) for i in x]

def process_data(filename):
    data['img_vec'] = data['img_vec'].apply(lambda x: x.strip('[').strip(']'))
    data['img_vec'] = data['img_vec'].apply(lambda x: x.split())
    data['img_vec'] = data['img_vec'].apply(lambda x: to_nums(x))
    data['sent_bert'] = data['sent_bert'].apply(lambda x: x.strip('[').strip(']'))
    data['sent_bert'] = data['sent_bert'].apply(lambda x: x.split(','))
    data['sent_bert'] = data['sent_bert'].apply(lambda x: to_nums(x))

def plot_with_matplotlib3(filename, column_name):
    data = pd.read_csv(filename)
    x_vals, y_vals, labels = reduce_dimensions2(data, column_name)
    meme_labels = []
    for x in labels:
      if x == 0:
        meme_labels.append('Non_meme')
      else:
        meme_labels.append('Meme')
    
    df = pd.DataFrame({'Reduced Dimension 1': x_vals, 
                       'Reduced Dimension 2': y_vals, 
                       'label': meme_labels})


    sns.scatterplot(data=df, x="Reduced Dimension 1", 
                    y="Reduced Dimension 2", 
                    hue='label',
                    alpha=0.8, 
                    palette="Set2")
    plt.title("T-SNE visualization of embeddings", fontsize=16)
    plt.xlabel("Reduced Dimension 1", fontsize=16)
    plt.ylabel("Reduced Dimension 2", fontsize=16)
    plt.show()

