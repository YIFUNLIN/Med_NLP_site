from django.shortcuts import render
from django.http import HttpResponse

def home(request):
    user_input = request.GET.get('input_text', '')  # 獲取用戶的輸入
    return render(request, 'index.html', {'output_text': user_input})  # 將用戶的輸入傳遞到模板中
