import numpy as np
from deepface import DeepFace
from apps.recognition.models.person import Person, FaceImage
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import tempfile

class FaceRecognitionService:
    def get_embedding(self, image_file):
        print("🔍 [1] Guardando imagen temporal...")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img = Image.open(image_file)
            img = img.convert("RGB")
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
                genero = analysis['gender']  # 'Man' o 'Woman'

                print(f"✅ Resultado - Edad: {edad}, Género: {genero}")
                return embedding, edad, genero

            except Exception as e:
                print("❌ Error en DeepFace:", e)
                return None

    def find_best_match(self, embedding, threshold=0.4):
        all_images = FaceImage.objects.select_related('person').all()
        best_match = None
        best_score = -1

        if embedding is None:
            return None

        for face_image in all_images:
            if face_image.embedding:
                stored_embedding = np.array(face_image.embedding)
                sim = cosine_similarity([embedding], [stored_embedding])[0][0]

                if sim > best_score and sim >= threshold:
                    best_score = sim
                    best_match = face_image.person

        return best_match
