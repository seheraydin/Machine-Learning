1- sitedenyorumcek.py dosyas�n� �al��t�rd���m�zda sitedeki verileri comment.csv dosyas�na �ekiyor
2- csv deki dosyalar� txt dosyas�na ta��yoruz.(data.txt)
3-sentimenAnalysis.py dosyas�n� �al��t�rd���m�zda data.txt deki verileri olu�turulmu� modele g�re 
   NEGAT�F-POZ�T�F OLARAK AYIRIP txt dosyalar�n� olu�turuyor
4- negatif.txt ve pozitif.pyp dosyalar�ndaki verileri �st taaraf pozitifler, alt tarak negatif ifadeler olmak �zere
    tumu.txt dosyas�nda birle�tiriyor. bu dosyay� csv dosyas�na �evirdikten sonra
5- yorumGir.py dosyas�n� yorum.csv dosyas�yla �al��t�r�p(ilk 200 sat�r olumlu kalan sat�rlar negatif olmak �zere verileri analiz ederiz)