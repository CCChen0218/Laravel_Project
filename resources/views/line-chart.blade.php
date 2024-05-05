<!DOCTYPE html>
<html>
<head>
    <title>Line Chart</title>
    <style>
        table {
            border-collapse: collapse;
            width: 50%;
        }
        th, td {
            text-align: center;
            padding: 8px;
        }
        th {
            background-color: #f2f2f2;
        }
        .warning {
            background-color: #ffcccc;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <canvas id="lineChart" width="800" height="400"></canvas>
    <table border="1">
        <thead>
            <tr>
                <th>Machine</th>
                @foreach ($labels as $label)
                    <th>{{ $label }}</th>
                @endforeach
                <th>low-yield-warning</th>
            </tr>
        </thead>
        <tbody>
            @foreach ($datasets as $dataset)
                <tr>
                    <td>{{ $dataset['label'] }}</td>
                    @foreach ($dataset['data'] as $key => $data)
                        <td>{{ $data }}</td>
                    @endforeach
                    <td class="{{ $dataset['data'][count($dataset['data']) - 1] < array_sum(array_slice($dataset['data'], 0, -1)) / (count($dataset['data']) - 1) ? 'warning' : '' }}">
                        {{ $dataset['data'][count($dataset['data']) - 1] < array_sum(array_slice($dataset['data'], 0, -1)) / (count($dataset['data']) - 1) ? 'Warning' : '' }}
                    </td>
                </tr>
            @endforeach
        </tbody>
    </table>
    <script>
        var labels = {!! json_encode($labels) !!};
        var datasets = {!! json_encode($datasets) !!};

        var ctx = document.getElementById('lineChart').getContext('2d');
        var lineChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: false,
                maintainAspectRatio: false,
            }
        });
    </script>
</body>
</html>

