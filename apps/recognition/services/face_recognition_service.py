def get_embedding(self, image_file):
    print("🔍 [1] Guardando imagen temporal...")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img = Image.open(image_file)
        img = img.convert("RGB")

        # 🔻 Reducción de tamaño para evitar bloqueos
        img = img.resize((320, 320))  # puedes probar también con 224x224

        img.save(tmp.name)

        try:
            print("📈 [2] Extrayendo embedding con DeepFace...")
            embedding_obj = DeepFace.represent(img_path=tmp.name, model_name='Facenet')[0]
            embedding = embedding_obj['embedding']

            print("📊 [3] Analizando edad y género...")
            analysis = DeepFace.analyze(
                img_path=tmp.name,
                actions=['age', 'gender']
            )[0]

            edad = int(analysis['age'])
            genero = analysis['gender']

            print(f"✅ Resultado - Edad: {edad}, Género: {genero}")
            return embedding, edad, genero

        except Exception as e:
            print("❌ Error en DeepFace:", e)
            return None
