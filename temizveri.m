
klasor_yolu = 'volunteer_7'; 


cikti_klasor_adi = ['temiz_csv_', klasor_yolu];


dat_dosyalari = [dir(fullfile(klasor_yolu, '**', '*.dat')); dir(fullfile(klasor_yolu, '**', '*.DAT'))];

total_dosya = length(dat_dosyalari);

if total_dosya == 0
    warning('HATA: Belirtilen klasörde veya alt klasörlerinde hiç .dat / .DAT dosyası bulunamadı!');
    fprintf('Lütfen sol paneldeki klasör adının "%s" ile birebir eşleştiğinden emin olun.\n', klasor_yolu);
else
    
    if ~exist(cikti_klasor_adi, 'dir')
        mkdir(cikti_klasor_adi);
    end

    fprintf('Toplam %d adet sinyal dosyası bulundu. Dönüştürme başlıyor...\n', total_dosya);

   
    for k = 1:total_dosya
        ham_dosya_adi = dat_dosyalari(k).name;
       
        ait_oldugu_klasor = dat_dosyalari(k).folder;
        tam_girdi_yolu = fullfile(ait_oldugu_klasor, ham_dosya_adi);

        fprintf('%d/%d işleniyor: %s\n', k, total_dosya, ham_dosya_adi);

        try
            
            samples = get_csi_amplitude_from_file(tam_girdi_yolu);

           
            A = reshape(samples, [], 30);

           
            [~, isim, ~] = fileparts(ham_dosya_adi);
            tam_cikti_yolu = fullfile(cikti_klasor_adi, [isim, '.csv']);

           
            writematrix(A, tam_cikti_yolu);

        catch ME
            fprintf('HATA: %s dosyası dönüştürülemedi. Geçiliyor...\n', ham_dosya_adi);
        end
    end

    disp(['TÜM SÜREÇ BİTTİ! Dosyalarınız "', cikti_klasor_adi, '" klasörüne kaydedildi.']);
end