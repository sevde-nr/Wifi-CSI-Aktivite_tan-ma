import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd


siniflar = [
    "bend", "drink_water", "draw_tick", "draw_x", "forward_kick", 
    "hand_clap", "high_arm_wave", "high_throw", "horizontal_arm_wave", 
    "phone_call", "side_kick", "sit_down", "squat", "toss_paper", 
    "two_hands_wave", "walk"
]


@st.cache_resource
def model_yukle():

    model_yolu = r"C:\Users\sevde\Desktop\Wifi_Project\wifiprocektphtyn\en_son_modelin.h5"
    return tf.keras.models.load_model(model_yolu)

try:
    model = model_yukle()
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu! Lütfen model_yolu alanını güncelleyin. Hata: {e}")


def csi_csv_hazirla(uploaded_file, sabit_uzunluk=700):
    try:
       
        df = pd.read_csv(uploaded_file, header=None)
        veri_numpy = df.to_numpy()
        
      
        if veri_numpy.shape[1] >= 15:
            veri_numpy = veri_numpy[:, 5:20]
        
       
        if veri_numpy.shape[0] < sabit_uzunluk:
            eksik = sabit_uzunluk - veri_numpy.shape[0]
            veri_numpy = np.pad(veri_numpy, ((0, eksik), (0, 0)), mode='constant')
        else:
            veri_numpy = veri_numpy[:sabit_uzunluk, :]
            
       
        veri_scaled = (
    veri_numpy - np.mean(veri_numpy)
) / (
    np.std(veri_numpy) + 1e-5
)
        
        
        girdi = np.expand_dims(veri_scaled, axis=0)
        return girdi, veri_numpy
        
    except Exception as e:
        st.error(f"Dosya işleme hatası: {e}")
        return None, None


st.set_page_config(
    page_title="Wi-Fi CSI Aktivite Analizi",
    page_icon="📡",
    layout="centered"
)

st.title("📡 Wi-Fi CSI ile Akıllı Ev ve Sağlık İzleme Sistemi")
st.info("Bilgisayar Mühendisliği Yapay Zeka Dersi Projesi — Sevde Nur Koç")


st.subheader("📁 Veri Analiz Merkezi")
yuklenen_dosya = st.file_uploader(
    "Temizlenmiş CSI (.csv) veri dosyasını seçin", 
    type=["csv"]
)


if yuklenen_dosya is not None:
   
    st.write(f"🔍 **Analiz Edilen Dosya:** `{yuklenen_dosya.name}`")
    
  
    model_girdisi, ham_grafik_verisi = csi_csv_hazirla(yuklenen_dosya)
    
    if model_girdisi is not None:
        
        tahmin_olasiliklari = model.predict(model_girdisi, verbose=0)[0]
        
       
        en_yuksek_indeks = np.argmax(tahmin_olasiliklari)
        tahmin_edilen_aktivite = siniflar[en_yuksek_indeks]
        guven_orani = tahmin_olasiliklari[en_yuksek_indeks] * 100
        
        st.divider()
        st.subheader("📊 Yapay Zeka Analiz Sonucu")
        
        
        if tahmin_edilen_aktivite in ["squat", "bend"]:
            st.warning(f"⚠️ ŞÜPHELİ DURUM: Model odadaki kişinin eğildiğini veya çömeldiğini tespit etti! ({tahmin_edilen_aktivite.upper()})")
            st.metric(label="Aktivite Güven Oranı", value=f"%{guven_orani:.2f}", delta="POTANSİYEL RİSK")
            
        elif tahmin_edilen_aktivite in ["walk", "drink_water", "phone_call", "sit_down"]:
            st.success(f"✅ RUTİN HAREKET: Odadaki kişi şu an normal günlük rutininde: {tahmin_edilen_aktivite.upper()}")
            st.metric(label="Aktivite Güven Oranı", value=f"%{guven_orani:.2f}", delta="GÜVENLİ", delta_color="inverse")
            
        else:
            st.info(f"ℹ️ AKTİVİTE TESPİT EDİLDİ: Odada şu an şu hareket yapılıyor: {tahmin_edilen_aktivite.upper()}")
            st.metric(label="Aktivite Güven Oranı", value=f"%{guven_orani:.2f}", delta="AKTİF")

        
        st.write("### 📈 Tüm Aktivite Olasılık Dağılımları")
        grafik_df = pd.DataFrame({
            'Aktiviteler': siniflar,
            'Olasılık Oranı (%)': tahmin_olasiliklari * 100
        }).sort_values(by='Olasılık Oranı (%)', ascending=False)
        
        st.bar_chart(data=grafik_df, x='Aktiviteler', y='Olasılık Oranı (%)')

       
        st.write("### 〰️ Yüklenen WiFi CSI Genlik Sinyali")
        st.caption("Filtrelenmiş sabit 15 kanaldan ilk 3 alt taşıyıcı frekansın zaman içerisindeki genlik değişimi:")
        
        
        grafik_sinyal_df = pd.DataFrame(ham_grafik_verisi[:, :3], columns=["Kanal 5", "Kanal 6", "Kanal 7"])
        st.line_chart(grafik_sinyal_df)

st.divider()
st.caption("Not: Bu sistem çok sınıflı 1D-CNN mimarisi ve 15 kanallı sabit WiFi frekans genlik matrisleri kullanılarak optimize edilmiştir.")