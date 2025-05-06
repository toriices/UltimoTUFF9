from django.db import models
from PIL import Image, UnidentifiedImageError
import numpy as np
import insightface

# Modelo singleton para evitar recarga innecesaria
class FaceModelSingleton:
    _instance = None

    @classmethod
    def get_model(cls):
        if cls._instance is None:
            cls._instance = insightface.app.FaceAnalysis(name='buffalo_l')
            try:
                cls._instance.prepare(ctx_id=-1, det_size=(640, 640))  # CPU por defecto
            except Exception as e:
                raise RuntimeError("Error al preparar el modelo de reconocimiento facial: " + str(e))
        return cls._instance

# 🧠 Modelo Person
class Person(models.Model):
    dni = models.CharField(max_length=20, unique=True)
    nombre = models.CharField(max_length=100)
    apellidos = models.CharField(max_length=100)
    edad = models.IntegerField(null=True, blank=True)
    genero = models.CharField(max_length=10, null=True, blank=True)

    def __str__(self):
        return f"{self.nombre} {self.apellidos} ({self.dni})"

# 🧠 Modelo FaceImage
class FaceImage(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='images')
    imagen = models.ImageField(upload_to='uploads/')
    embedding = models.JSONField(null=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.embedding and self.imagen:
            try:
                image = Image.open(self.imagen)
                image = image.convert('RGB')
                image = image.resize((640, 640))  # Redimensionar para evitar problemas
                image_np = np.array(image)

                face_model = FaceModelSingleton.get_model()
                faces = face_model.get(image_np)

                if faces:
                    self.embedding = faces[0].embedding.tolist()
                else:
                    # Embedding se mantiene como None, pero no se interrumpe el guardado
                    print("⚠️ Advertencia: No se detectó ninguna cara en la imagen subida.")

            except UnidentifiedImageError:
                print("❌ Error: No se pudo identificar la imagen. Asegúrate de que sea un archivo válido.")
            except Exception as e:
                print(f"❌ Error procesando la imagen: {str(e)}")

        super().save(*args, **kwargs)

    def __str__(self):
        return f"Imagen de {self.person.nombre} {self.person.apellidos}"
