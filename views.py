from django.shortcuts import render
from .models import CrowdData
from .crowd_processing import process_image
from .csrnet import csrnet_model



def home(request):
    if request.method == 'POST':
        image = request.FILES['image']

        num_people, datetime_taken, image_np = process_image(csrnet_model, image)
        crowd_data = CrowdData(people_count=num_people, date_taken=datetime_taken, image=image)
        crowd_data.save()
        datetime_taken_str = datetime_taken.strftime("%Y-%m-%dT%H:%M:%S.%f")

        response_data = {
            'num_people': float(num_people),
            'time_taken': datetime_taken_str,
            'date_taken': datetime_taken_str,
        }
        return JsonResponse(response_data)

    else:
        crowd_data = CrowdData.objects.all()
        return render(request, 'index.html', {'crowd_data': crowd_data})
