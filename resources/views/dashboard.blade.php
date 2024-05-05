<link href="{{ asset('css/custom.css') }}" rel="stylesheet" type="text/css">
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-wEmeIV1mKuiNpC+IOBjI7aAzPcEZeedi5yW5f2yOq55WWLwNGmvvx4Um1vskeMj0" crossorigin="anonymous">
<x-app-layout>
    <x-slot name="header">
        <h2 class="font-semibold text-xl text-gray-800 leading-tight">
            {{ __('Dashboard') }}
        </h2>
    </x-slot>

    <div class="py-12">
        <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
            <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg">
                <div class="p-6 text-gray-900">
                    {{ __("You're logged in!") }}
                    <div><hr><br></div>
                    <form id="upload-form" action="{{ route('upload') }}" method="POST" enctype="multipart/form-data">
                        @csrf
                        <input type="file" name="file" id="file-input" style="display: none;" accept=".csv;">
                        <button type="button" onclick="document.getElementById('file-input').click();" class="btn btn-outline-primary">Choose File</button>
                        <span id="file-name">No file chosen</span>
                        <button type="submit" class="btn btn-outline-primary">Upload</button>
                        <button type="button" id="process-btn" class="btn btn-outline-primary">Process</button>
                        <button type="button" id="show-chart-btn" class="btn btn-outline-primary" onclick="window.location.href='{{ route('LineChart') }}'">Chart</button>
                        <style>
                            .btn-outline-primary {
                                background-color: #007bff;
                                color: #fff
                            }
                        </style>
                    </form>
                    <div id="upload-message" class="alert alert-success"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.querySelector('#upload-form');

            form.addEventListener('submit', function(event) {
                event.preventDefault();

                const formData = new FormData(form);

                fetch(form.getAttribute('action'), {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.querySelector('#upload-message').innerText = data.message;
                    document.querySelector('#upload-message').style.display = 'block';
                })
                .catch(error => console.error('Error:', error));
            });

            const fileInput = document.getElementById('file-input');
            const fileNameSpan = document.getElementById('file-name');

            fileInput.addEventListener('change', function() {
                if (fileInput.files.length > 0) {
                    fileNameSpan.innerText = fileInput.files[0].name;
                } else {
                    fileNameSpan.innerText = 'No file chosen';
                }
            });

            const processBtn = document.getElementById('process-btn');

            processBtn.addEventListener('click', function() {
                fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRF-TOKEN': '{{ csrf_token() }}'
                    }
                })
                .then(response => response.json())
                .then(data => {
                    document.querySelector('#upload-message').innerText = data.message;
                    document.querySelector('#upload-message').style.display = 'block';
                    console.log(data);
                })
                .catch(error => console.error('Error:', error));
            });

        });
    </script>
</x-app-layout>