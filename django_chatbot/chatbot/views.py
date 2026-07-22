import base64
import os
import re

from dotenv import load_dotenv
from django.contrib import auth
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.utils import IntegrityError
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from openai import OpenAI, OpenAIError

from .models import Chat

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
openai_base_url = os.getenv('OPENAI_BASE_URL', 'https://api.groq.com/openai/v1')
text_model = os.getenv('AI_TEXT_MODEL', 'openai/gpt-oss-20b')
vision_model = os.getenv('AI_VISION_MODEL', 'qwen/qwen3.6-27b')
client = OpenAI(api_key=openai_api_key, base_url=openai_base_url) if openai_api_key else None

ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/webp', 'image/gif'}
MAX_IMAGE_SIZE = 20 * 1024 * 1024


def image_to_data_url(image):
    if image.size > MAX_IMAGE_SIZE:
        raise ValueError('Image is too large. Maximum size is 20 MB.')

    content_type = image.content_type or 'application/octet-stream'
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError('Unsupported image type. Upload JPG, PNG, WEBP, or GIF.')

    encoded_image = base64.b64encode(image.read()).decode('utf-8')
    return f'data:{content_type};base64,{encoded_image}'


def ask_openai(message, image=None):
    if client is None or openai_api_key == 'your_api_key_here':
        return 'API key is not configured. Add your real key to the .env file.'

    try:
        if image:
            data_url = image_to_data_url(image)
            user_text = message or 'Describe this image and answer in the user language.'
            response = client.chat.completions.create(
                model=vision_model,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant. Answer clearly and use Markdown when useful.'},
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': user_text},
                            {'type': 'image_url', 'image_url': {'url': data_url}},
                        ],
                    },
                ],
                max_completion_tokens=2048,
            )
        else:
            response = client.chat.completions.create(
                model=text_model,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant. Answer clearly and use Markdown when useful.'},
                    {'role': 'user', 'content': message},
                ],
            )
    except ValueError as error:
        return str(error)
    except OpenAIError as error:
        return f'AI API error: {error}'

    content = response.choices[0].message.content
    if content:
        # Strip <think>...</think> tags and everything between them
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # Strip any unclosed <think> tag if the response was truncated
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
        content = content.strip()
    return content


@login_required(login_url='login')
def chatbot(request):
    chats = Chat.objects.filter(user=request.user)

    if request.method == 'POST':
        message = request.POST.get('message', '').strip()
        image = request.FILES.get('image')

        if not message and not image:
            return JsonResponse({'error': 'Message or image is required.'}, status=400)

        response = ask_openai(message, image)
        saved_message = message if message else '[Image]'
        if image:
            saved_message = f'{saved_message}\n[Attached image: {image.name}]'

        chat = Chat(user=request.user, message=saved_message, response=response, created_at=timezone.now())
        chat.save()
        return JsonResponse({'message': saved_message, 'response': response})
    return render(request, 'chatbot.html', {'chats': chats})


def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('chatbot')
        else:
            error_message = 'Invalid username or password'
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')


def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 == password2:
            try:
                user = User.objects.create_user(username, email, password1)
                user.save()
                auth.login(request, user)
                return redirect('chatbot')
            except IntegrityError:
                error_message = 'Error creating account'
                return render(request, 'register.html', {'error_message': error_message})
        else:
            error_message = 'Password dont match'
            return render(request, 'register.html', {'error_message': error_message})
    return render(request, 'register.html')


def logout(request):
    auth.logout(request)
    return redirect('login')