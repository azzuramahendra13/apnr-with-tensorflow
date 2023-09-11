CREATE TABLE log(
    id_log INT,
    nomor_kendaraan VARCHAR(10),
    waktu_masuk DATETIME,
    waktu_keluar DATETIME,
    pemilik VARCHAR(20),
    status_kendaraan VARCHAR(20)

    CONSTRAINT PK_ID_log PRIMARY KEY (id_log)
);


