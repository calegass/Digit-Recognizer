import sys

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# Definindo constantes
IMAGE_DIM = 28
CHANNELS = 1
CONVOLUTION_MASK = 3
POOLING_MASK = 2
BATCH_SIZE = 86

# Configurações de estilo do seaborn
sns.set_theme(style='white', context='notebook', palette='deep')


def check_environment():
    """Verifica e imprime o ambiente de execução."""
    print(sys.executable)


@st.cache_data
def load_data(train_path='data/train.csv', test_path='data/test.csv'):
    """Carrega os dados de treino e teste a partir dos arquivos CSV."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


@st.cache_data
def preprocess_data(train, test):
    """Pré-processa os dados de treino e teste."""
    Y_train = train['label']
    X_train = train.drop(labels=['label'], axis=1)

    # Normalizando e reshaping dos dados
    X_train = X_train / 255.0
    test = test / 255.0

    X_train = X_train.values.reshape(-1, IMAGE_DIM, IMAGE_DIM, CHANNELS)
    test = test.values.reshape(-1, IMAGE_DIM, IMAGE_DIM, CHANNELS)

    # One-hot encoding das labels
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=10)

    # Separando dados de treino e validação
    train_size = int(X_train.shape[0] * 0.8)
    X_train, X_val = X_train[:train_size], X_train[train_size:]
    Y_train, Y_val = Y_train[:train_size], Y_train[train_size:]

    return X_train, Y_train, X_val, Y_val, test


def build_model(filters):
    """Cria e retorna o modelo de rede neural convolucional."""
    model = tf.keras.Sequential()

    # Adicionando camadas convolucionais
    for i, filter_ in enumerate(filters):
        # Primeira camada convolucional precisa definir input_shape
        if i == 0:
            model.add(tf.keras.layers.Conv2D(filter_, (CONVOLUTION_MASK, CONVOLUTION_MASK),
                                             input_shape=(IMAGE_DIM, IMAGE_DIM, CHANNELS),
                                             activation='relu', kernel_initializer='he_normal', padding='same'))
        else:
            model.add(tf.keras.layers.Conv2D(filter_, (CONVOLUTION_MASK, CONVOLUTION_MASK),
                                             activation='relu', kernel_initializer='he_normal', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(filter_, (CONVOLUTION_MASK, CONVOLUTION_MASK),
                                         activation='relu', kernel_initializer='he_normal', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(POOLING_MASK, POOLING_MASK)))
        model.add(tf.keras.layers.Dropout(0.3))

    # Adicionando camadas densas
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    # Compilando o modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, X_train, Y_train, X_val, Y_val, epochs, batch_size):
    """Treina o modelo de rede neural com os dados de treino e validação."""

    # # Callbacks desativados já que a lógica não funciona de época em época
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1,
    #                                                   restore_best_weights=True)
    # learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
    #                                                                verbose=1, min_lr=0.00001)

    # Data augmentation
    train_data = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10, zoom_range=0.1,
                                                                 width_shift_range=0.1, height_shift_range=0.1)
    train_data.fit(X_train)

    # Treinamento do modelo
    history = []
    for epoch in range(epochs):
        # Treinando por uma época
        print(f'Epoch {epoch + 1}/{epochs}')
        history_epoch = model.fit(train_data.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                  epochs=1, validation_data=(X_val, Y_val),
                                  steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                                  # callbacks=[early_stopping, learning_rate_reduction]
                                  )

        # Atualizando as métricas para cada época
        loss = history_epoch.history['loss'][0]
        val_loss = history_epoch.history['val_loss'][0]
        accuracy = history_epoch.history['accuracy'][0]
        val_accuracy = history_epoch.history['val_accuracy'][0]

        history.append((loss, val_loss, accuracy, val_accuracy))

        # Exibindo as métricas na interface do Streamlit
        st.markdown("<br>", unsafe_allow_html=True)
        st.write(f'**Época {epoch + 1}/{epochs}** - '
                 f'**Perda:** {loss:.4f} - **Perda (Validação):** {val_loss:.4f} - '
                 f'**Acurácia:** {accuracy:.4f} - **Acurácia (Validação):** {val_accuracy:.4f}')

    return history


def plot_accuracy_loss(history):
    """Plota a acurácia e a perda do treinamento e validação."""

    # Separando perdas e acurácias
    losses, val_losses, accuracies, val_accuracies = zip(*history)

    # Plotando a acurácia
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(accuracies, label='Acurácia de Treinamento')
    ax[0].plot(val_accuracies, label='Acurácia de Validação')
    ax[0].set_title('Acurácia do Modelo')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Acurácia')
    ax[0].legend()

    # Plotando a perda
    ax[1].plot(losses, label='Perda de Treinamento')
    ax[1].plot(val_losses, label='Perda de Validação')
    ax[1].set_title('Perda do Modelo')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('Perda')
    ax[1].legend()

    plt.tight_layout()
    st.pyplot(fig)


def plot_confusion_matrix(model, X_val, Y_val):
    """Plota a matriz de confusão para os dados de validação."""
    # Predições do modelo
    Y_pred = model.predict(X_val)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_val, axis=1)

    # Gerando a matriz de confusão
    cm = confusion_matrix(Y_true, Y_pred_classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=np.arange(10), yticklabels=np.arange(10), ax=ax)
    ax.set_title('Matriz de Confusão')
    ax.set_xlabel('Classe Prevista')
    ax.set_ylabel('Classe Verdadeira')

    st.pyplot(fig)


def show_sample_errors(model, X_val, Y_val, num_samples):
    """Exibe algumas amostras onde o modelo errou."""
    # Predições do modelo
    Y_pred = model.predict(X_val)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_val, axis=1)

    # Encontrando os índices dos erros
    errors = np.where(Y_pred_classes != Y_true)[0]

    if len(errors) == 0:
        st.write("Nenhum erro encontrado nas previsões!")
        return
    else:
        st.write(f"Encontrados {len(errors)} erros nas previsões.")

    # Selecionando um número específico de erros
    error_indices = np.random.choice(errors, size=min(num_samples, len(errors)), replace=False)

    # Plotando as amostras com erros
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 6))

    for i, index in enumerate(error_indices):
        axes[i].imshow(X_val[index].reshape(IMAGE_DIM, IMAGE_DIM), cmap='gray')
        axes[i].set_title(f'Previsto: {Y_pred_classes[index]}\nReal: {Y_true[index]}')
        axes[i].axis('off')

    st.pyplot(fig)


# Main function for Streamlit
def main():
    st.title("Reconhecimento de Dígitos com CNN")
    st.write(
        "Esta aplicação permite treinar um modelo de rede neural convolucional para reconhecer dígitos com base no dataset: https://www.kaggle.com/c/digit-recognizer/data"
        ". Carregue os dados e ajuste os parâmetros conforme necessário.")

    # Carregar dados
    st.subheader("Carga de Dados")
    train, test = load_data()
    st.write("Dados de treino e teste carregados com sucesso!")

    # Pré-processar dados
    st.subheader("Pré-processamento de Dados")
    X_train, Y_train, X_val, Y_val, test = preprocess_data(train, test)
    st.write("Dados pré-processados com sucesso!")

    # Construir o modelo
    st.subheader("Construção do Modelo")
    filters = st.multiselect('Selecione os filtros para as camadas convolucionais:', options=[32, 64, 128],
                             default=[32, 64])
    batch_size = st.slider("Tamanho do Lote (Batch Size)", min_value=32, max_value=128, value=BATCH_SIZE, step=32)

    # Treinamento do modelo
    st.subheader("Treinamento do Modelo")
    epochs = st.slider("Número de épocas", min_value=1, max_value=50, value=10)

    # Inicializar modelo se filtros forem selecionados
    if filters:
        if 'model' not in st.session_state:
            model = build_model(filters)
            st.session_state.model = model
            st.write("Modelo construído com sucesso!")
        else:
            model = st.session_state.model
            st.write("Modelo construído com sucesso!")

    if st.button("Treinar Modelo"):
        # Treinar apenas se ainda não houver histórico salvo
        if 'history' not in st.session_state:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.spinner("Treinando o modelo..."):
                st.write("Logs em tempo real:")
                history = train_model(model, X_train, Y_train, X_val, Y_val, epochs, batch_size)
                st.session_state.history = history
            st.markdown("<br>", unsafe_allow_html=True)
            st.success("Modelo treinado com sucesso!")
        else:
            st.warning("Modelo já foi treinado. Reinicie a aplicação para treinar novamente.")

    # Exibir resultados se o treinamento foi realizado
    if 'history' in st.session_state:
        st.subheader("Resultados do Treinamento")
        plot_accuracy_loss(st.session_state.history)

        # Plotar matriz de confusão
        st.subheader("Matriz de Confusão")
        plot_confusion_matrix(model, X_val, Y_val)

        # Mostrar amostras de erro
        st.subheader("Amostras de Erros")
        num_samples = st.number_input("Número de amostras para mostrar:", min_value=1, max_value=20, value=5)
        show_sample_errors(model, X_val, Y_val, num_samples)


if __name__ == "__main__":
    main()
