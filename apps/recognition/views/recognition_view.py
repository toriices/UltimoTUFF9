from django.shortcuts import render
from apps.recognition.models.person import Person, FaceImage
from PIL import Image
import numpy as np
from apps.recognition.models.person import FaceModelSingleton
from apps.recognition.services.face_recognition_service import FaceRecognitionService

face_service = FaceRecognitionService()

def register_person(request):
    if request.method == 'POST':
        dni = request.POST.get('dni')
        nombre = request.POST.get('nombre')
        apellidos = request.POST.get('apellidos')
        foto = request.FILES.get('foto')

        if not foto:
            return render(request, 'recognition/register.html', {'error': 'Se requiere una foto'})

        if Person.objects.filter(dni=dni).exists():
            return render(request, 'recognition/register.html', {'error': 'El DNI ya está registrado'})

        person = Person.objects.create(dni=dni, nombre=nombre, apellidos=apellidos)

        try:
            # Procesar imagen
            image = Image.open(foto).convert('RGB').resize((640, 640))
            image_np = np.array(image)

            model = FaceModelSingleton.get_model()
            faces = model.get(image_np)

            if not faces:
                person.delete()
                return render(request, 'recognition/register.html', {'error': 'No se detectó ninguna cara'})

            embedding = faces[0].embedding.tolist()
            FaceImage.objects.create(person=person, imagen=foto, embedding=embedding)

        except Exception as e:
            person.delete()
            return render(request, 'recognition/register.html', {'error': f'Error procesando imagen: {e}'})

        return render(request, 'recognition/person_registered.html', {'person': person})

    return render(request, 'recognition/register.html')


def upload_and_recognize(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']

        result = face_service.get_embedding(image)
        
        if result is not None:
            embedding, edad, genero = result
            person = face_service.find_best_match(embedding)

            if person:
                if person.edad is None:
                    person.edad = edad
                if person.genero is None:
                    person.genero = genero
                person.save()

                face_image = FaceImage.objects.filter(person=person).first()

                context['recognized'] = {
                    'nombre': person.nombre,
                    'apellidos': person.apellidos,
                    'dni': person.dni,
                    'edad': person.edad,
                    'genero': person.genero,
                    'foto': face_image.imagen.url if face_image else None,
                }
            else:
                context['error'] = "No se encontró coincidencia."
        else:
            context['error'] = "No se detectó ninguna cara en la imagen subida."

    return render(request, 'recognition/upload.html', context)
