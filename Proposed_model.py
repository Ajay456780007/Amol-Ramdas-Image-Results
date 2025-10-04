import joblib
import numpy as np
import lightgbm as lgb
from keras.optimizers import Adam
from keras_facenet import FaceNet
from tensorflow.keras import layers, models
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from Sub_Functions.Evaluate import main_est_parameters
from Sub_Functions.Load_data import train_test_splitter


def build_attention_model(input_shape, num_classes):
    input_layer = layers.Input(shape=input_shape)  # shape (512,)

    x = layers.Reshape((1, input_shape[0]))(input_layer)  # (1, 512)
    x = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def proposed_model(x_train, x_test, y_train, y_test, train_percent, DB, s=False):
    import os
    import numpy as np
    import joblib
    import lightgbm as lgb
    from keras.utils import to_categorical, plot_model
    from sklearn.preprocessing import LabelEncoder
    from keras.optimizers import Adam
    from keras import models

    # Step 1: Extract FaceNet Embeddings
    facenet = FaceNet()
    x_train_emb = facenet.embeddings(x_train)  # shape: (N, 512)
    x_test_emb = facenet.embeddings(x_test)    # shape: (M, 512)
    input_shape = x_train_emb.shape[1:]

    # Step 2: Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    y_train_cat = to_categorical(y_train_enc)
    y_test_cat = to_categorical(y_test_enc)
    num_classes = y_train_cat.shape[1]

    # Step 3: Directory setup
    Checkpoint_dir = f"Checkpoint/{DB}/TP_{int(train_percent * 100)}"
    os.makedirs(Checkpoint_dir, exist_ok=True)

    metric_path = f"Analysis/Performance_Analysis/{DB}/"
    os.makedirs(metric_path, exist_ok=True)

    os.makedirs("Architectures/", exist_ok=True)
    # os.makedirs("Saved_model/", exist_ok=True)

    # Step 4: Training loop with checkpoints
    base_epochs = [100, 200, 300, 400, 500]
    prev_epoch = 0
    metrics_all = {}

    attention_model = build_attention_model(input_shape, num_classes)
    attention_model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    for ep in reversed(base_epochs):
        ckt_path = os.path.join(Checkpoint_dir, f"attention_model_epoch_{ep}.weights.h5")
        metrics_path = os.path.join(metric_path, f"facenet_attention_metrics_{train_percent}percent_epoch{ep}.npy")
        if os.path.exists(ckt_path) and os.path.exists(metrics_path):
            print(f"Found existing full checkpoint and metrics for epoch {ep}, loading and resuming...")
            attention_model.load_weights(ckt_path)
            prev_epoch = ep
            break

    for end_epochs in base_epochs:
        if end_epochs <= prev_epoch:
            continue

        print(f"\nTraining from epoch {prev_epoch + 1} to {end_epochs} for TP={train_percent * 100}%...")

        ckt_path = os.path.join(Checkpoint_dir, f"attention_model_epoch_{end_epochs}.weights.h5")
        metrics_path = os.path.join(metric_path, f"facenet_attention_metrics_{train_percent}percent_epoch{end_epochs}.npy")

        try:
            attention_model.fit(
                x_train_emb, y_train_cat,
                epochs=end_epochs,
                initial_epoch=prev_epoch,
                batch_size=32,
                validation_split=0.1,
                verbose=1
            )

            plot_model(attention_model, to_file="Architectures/attention_model.png", show_shapes=True, show_layer_names=True)
            attention_model.save_weights(ckt_path)
            print(f"Checkpoint saved at: {ckt_path}")
            if s:
                attention_model.save(f"Saved_model/{DB}_attention_model.h5")

            # Step 5: Refined embeddings from penultimate layer
            embedding_model = models.Model(inputs=attention_model.input, outputs=attention_model.layers[-2].output)
            train_embeddings = embedding_model.predict(x_train_emb)
            test_embeddings = embedding_model.predict(x_test_emb)

            # Step 6: LightGBM classifier
            lgbm_model = lgb.LGBMClassifier(verbose=-1)
            lgbm_model.fit(train_embeddings, y_train_enc)

            if s:
                joblib.dump(lgbm_model, f"Saved_model/{DB}_lgbm_model_epoch{end_epochs}.pkl")

            y_pred_lgbm = lgbm_model.predict(test_embeddings)

            # Step 7: Evaluate
            metrics = main_est_parameters(y_test_enc, y_pred_lgbm)
            metrics_all[f"epoch_{end_epochs}"] = metrics
            np.save(metrics_path, metrics)
            print(f"Metrics saved at: {metrics_path}")

            prev_epoch = end_epochs

        except KeyboardInterrupt:
            print(f"Training interrupted during epoch chunk {prev_epoch + 1}–{end_epochs}. Not saving checkpoint or metrics.")
            raise

        print(f"\nCompleted training for {int(train_percent*100)}% up to {prev_epoch} epochs.")

    return metrics_all

