import face_recognition
import numpy as np
import tempfile
from PIL import Image
from deepface import DeepFace
from apps.recognition.models.person import Person, FaceImage
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognitionService:

    def get_embedding(self, image_file):
        print("🔍 [1] Procesando imagen con face_recognition...")
        try:
            # Preparar imagen
            img = Image.open(image_file)
            img = img.convert("RGB")
            img = img.resize((640, 640))

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img.save(tmp.name)
                image_np = face_recognition.load_image_file(tmp.name)

            # Embedding
            face_locations = face_recognition.face_locations(image_np, model='cnn')
            if not face_locations:
                print("❌ No se detectó ninguna cara.")
                return None

            encodings = face_recognition.face_encodings(image_np, face_locations)
            if not encodings:
                print("❌ No se pudo calcular el encoding.")
                return None

            # [🧠] Predicción real de edad y género
            print("📊 [2] Obteniendo edad y género reales...")
            analysis = DeepFace.analyze(
                img_path=tmp.name,
                actions=["age", "gender"],
                enforce_detection=False
            )[0]


            edad = int(analysis["age"])
            genero = analysis["gender"]

            print(f"✅ Embedding OK | Edad: {edad}, Género: {genero}")
            return encodings[0].tolist(), edad, genero

        except Exception as e:
            print("❌ Error procesando imagen:", e)
            return None

    def find_best_match(self, embedding, threshold=0.4):
        all_images = FaceImage.objects.select_related('person').all()
        best_match = None
        best_score = -1

        if embedding is None:
            return None

        print(f"\n🔎 Comparando con {len(all_images)} imágenes guardadas...\n")

        for face_image in all_images:
            if face_image.embedding:
                stored_embedding = np.array(face_image.embedding)

                print(f"↪️ {face_image.person.nombre} {face_image.person.apellidos}")
                print(f"   🧬 Dimensiones: actual={len(embedding)} vs guardado={stored_embedding.shape[0]}")

                if stored_embedding.shape[0] != len(embedding):
                    print("   ⚠️ Ignorado por longitud incompatible.\n")
                    continue

                sim = cosine_similarity([embedding], [stored_embedding])[0][0]
                print(f"   📊 Similitud: {sim:.4f}")

                if sim > best_score and sim >= threshold:
                    best_score = sim
                    best_match = face_image.person
                    print("   ✅ ¡Posible mejor coincidencia!\n")
                else:
                    print("   ❌ Debajo del umbral\n")

        if best_match:
            print(f"🎯 Coincidencia final: {best_match.nombre} con similitud {best_score:.4f}")
        else:
            print("❌ No se encontró ninguna coincidencia suficientemente similar.")

        return best_match
