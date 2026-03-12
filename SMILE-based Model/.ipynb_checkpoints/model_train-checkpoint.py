# model_train.py
"""
Model building & training utilities for MOA classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight


def build_mlp_model(input_shape, num_classes):
    """
    Build a simple feedforward MLP for MOA prediction.

    Parameters
    ----------
    input_shape : tuple
        Input shape of features, e.g. (2048,)
    num_classes : int
        Number of MOA classes

    Returns
    -------
    keras.Model
        Compiled MLP model
    """
    input_layer = Input(shape=input_shape)
    x = Dense(64, activation='relu')(input_layer)
    x = Dropout(0.75)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def train_mlp_model(model, x_train, y_train, x_val, y_val, class_weights_dict,
                    epochs=1000, batch_size=64):
    """
    Train an MLP with early stopping & LR reduction.

    Parameters
    ----------
    model : keras.Model
        Model from build_mlp_model
    x_train, y_train : np.ndarray
        Training features and labels
    x_val, y_val : np.ndarray
        Validation features and labels
    class_weights_dict : dict
        Class weights for imbalance handling
    epochs : int
        Max epochs
    batch_size : int
        Batch size

    Returns
    -------
    (model, history)
        Trained model and Keras History object
    """
    earlyStopping = EarlyStopping(monitor='val_loss', patience=25, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=7, min_delta=1e-119, mode='min')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        class_weight=class_weights_dict,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[earlyStopping, reduce_lr],
        verbose=1
    )

    return model, history


def plot_training_history(history, output_path="training_history.png"):
    """
    Plot training and validation accuracy/loss curves.

    Parameters
    ----------
    history : keras.callbacks.History
        Training history object
    output_path : str
        Where to save plot
    """
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_moa_bar(df, output_path="moa_distribution_barplot.png"):
    """
    Plot number of compounds per MOA class, separated by train/val/test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'moa_label' and 'set' columns.
    output_path : str
        File path to save barplot.

    Notes
    -----
    - Bars are grouped by MOA.
    - Colors:
        train = cornflowerblue
        val   = peru
        test  = palevioletred
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    counts = (
        df.groupby(["moa_label", "set"])
          .size()
          .reset_index(name="count")
    )

    plt.figure(figsize=(10, 6))
    hue_order = ["tr", "val", "test"]
    palette = {"tr": "cornflowerblue", "val": "peru", "test": "palevioletred"}
    
    sns.barplot(
        data=counts,
        x="moa_label",
        y="count",
        hue="set",
        hue_order=hue_order,
        palette=palette)
    
    plt.legend(
        title="Set",
        labels=["Train", "Validation", "Test"],
        title_fontsize=12,
        fontsize=10)


def run_moa_training_pipeline(clustered_df, model_save_path, label_dict):
    """
    Run training given a pre-split clustered dataset.

    Parameters
    ----------
    clustered_df : pd.DataFrame
        Must include ['moa_label','morgan_fps','set']
    model_save_path : str
        Path to save trained model
    label_dict : dict
        Precomputed mapping {MOA: integer label}

    Returns
    -------
    model : keras.Model
        Trained model
    history : keras.callbacks.History
        Training history
    """
    clustered_df['labels'] = clustered_df['moa_label'].map(label_dict)

    from data_splitting import prepare_data
    x_tr, y_tr, x_val, y_val, x_test, y_test = prepare_data(
        clustered_df[['moa_label', 'morgan_fps', 'set', 'labels']]
    )

    from sklearn.utils.class_weight import compute_class_weight
    class_weights_dict = dict(enumerate(
        compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    ))

    model = build_mlp_model((x_tr.shape[1],), len(label_dict))
    model, history = train_mlp_model(model, x_tr, y_tr, x_val, y_val, class_weights_dict)

    return model, history
