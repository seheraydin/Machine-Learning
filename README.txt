1- sitedenyorumcek.py dosyasýný çalýþtýrdýðýmýzda sitedeki verileri comment.csv dosyasýna çekiyor
2- csv deki dosyalarý txt dosyasýna taþýyoruz.(data.txt)
3-sentimenAnalysis.py dosyasýný çalýþtýrdýðýmýzda data.txt deki verileri oluþturulmuþ modele göre 
   NEGATÝF-POZÝTÝF OLARAK AYIRIP txt dosyalarýný oluþturuyor
4- negatif.txt ve pozitif.pyp dosyalarýndaki verileri üst taaraf pozitifler, alt tarak negatif ifadeler olmak üzere
    tumu.txt dosyasýnda birleþtiriyor. bu dosyayý csv dosyasýna çevirdikten sonra
5- yorumGir.py dosyasýný yorum.csv dosyasýyla çalýþtýrýp(ilk 200 satýr olumlu kalan satýrlar negatif olmak üzere verileri analiz ederiz)