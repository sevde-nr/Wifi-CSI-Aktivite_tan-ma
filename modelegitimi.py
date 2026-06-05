import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


EPOCHS = 80             
BATCH_SIZE = 32
SABIT_UZUNLUK = 700
SINIF_SAYISI = 16

aktivite_haritasi = {
    1: "horizontal_arm_wave", 2: "high_arm_wave", 3: "two_hands_wave", 4: "high_throw",
    5: "draw_x", 6: "draw_tick", 7: "toss_paper", 8: "forward_kick",
    9: "side_kick", 10: "bend", 11: "hand_clap", 12: "walk",
    13: "phone_call", 14: "drink_water", 15: "sit_down", 16: "squat"
}


def veri_setini_yukle_akilli_filtreli(data_klasor_yolu):
    X_list = []
    y_list = []
    
    csv_dosyalari = glob.glob(os.path.join(data_klasor_yolu, "**", "*.csv"), recursive=True)
    print(f"Toplam {len(csv_dosyalari)} adet CSV dosyası bulundu. Akıllı filtreyle yükleniyor...")

    if len(csv_dosyalari) == 0:
        return np.array([]), np.array([])

    for dosya in csv_dosyalari:
        dosya_adi = os.path.basename(dosya)
        try:
            parcalar = dosya_adi.split('_')
            aktivite_kodu = parcalar[1]
            aktivite_no = int(aktivite_kodu[1:])
            
            if aktivite_no in aktivite_haritasi:
                df = pd.read_csv(dosya, header=None)
                veri_numpy = df.to_numpy()
                
                
                if veri_numpy.shape[1] >= 15:
                    veri_numpy = veri_numpy[:, 5:20] 
                
                if veri_numpy.shape[0] < SABIT_UZUNLUK:
                    eksik = SABIT_UZUNLUK - veri_numpy.shape[0]
                    veri_numpy = np.pad(veri_numpy, ((0, eksik), (0, 0)), mode='constant')
                else:
                    veri_numpy = veri_numpy[:SABIT_UZUNLUK, :]
                
                X_list.append(veri_numpy)
                y_list.append(aktivite_haritasi[aktivite_no])
                
        except Exception as e:
            continue
            
    return np.array(X_list), np.array(y_list)


klasor_yolu = r"C:\Users\sevde\Desktop\Wifi_Project\wifiprocektphtyn\wiar_data"
X_raw, y_raw = veri_setini_yukle_akilli_filtreli(klasor_yolu)

if X_raw.size == 0:
    print("HATA: Dosyalar yüklenemedi!")
else:
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    
    
    X_scaled = (X_raw - np.mean(X_raw, axis=(1,2), keepdims=True)) / \
           (np.std(X_raw, axis=(1,2), keepdims=True) + 1e-5) 
    np.save(r"C:\Users\sevde\Desktop\Wifi_Project\wifiprocektphtyn\siniflar.npy", le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print("Veri örüntüsü başarıyla hizalandı. Model mimarisine geçiliyor...")


    giris_boyutu = (X_train.shape[1], X_train.shape[2]) 

    model = Sequential([
        Conv1D(filters=64, kernel_size=9, activation='relu', input_shape=giris_boyutu),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Flatten(),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(SINIF_SAYISI, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()


    model_kayit_yolu = r"C:\Users\sevde\Desktop\Wifi_Project\wifiprocektphtyn\en_son_model.h5"
    
   
    checkpoint = ModelCheckpoint(
        filepath=model_kayit_yolu,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )


    print("\n--- Model Eğitimi Başlıyor ---")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop], 
        verbose=1
    )
    print("\nModel Eğitimi Başarıyla Tamamlandı!")


    print("\n--- Kaydedilen En Kararlı Altın Model Yükleniyor ve Test Ediliyor ---")
    en_iyi_model = tf.keras.models.load_model(model_kayit_yolu)
    
    
    y_pred_probs = en_iyi_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    
    print("\n================ DETAYLI AKADEMİK BAŞARI RAPORU ================")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("=================================================================")

    
    print("\n================ DOĞRULUK MATRİSİ (CONFUSION MATRIX) ================")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df)
    print("=====================================================================")

    
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu (Train Acc)')
    plt.plot(history.history['val_accuracy'], label='Test Doğruluğu (Val Acc)')
    plt.title('Epochs Boyunca Model Başarı Değişimi')
    plt.xlabel('Epoch Sayısı')
    plt.ylabel('Doğruluk (Accuracy)')
    plt.legend()
    plt.grid(True)
    plt.show()
