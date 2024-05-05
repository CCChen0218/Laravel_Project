<!DOCTYPE html>
<html>
<head>
    <title>Line Chart</title>
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
                <th>Max</th> <!-- 新添加的列 -->
            </tr>
        </thead>
        <tbody>
            @foreach ($datasets as $dataset)
                <tr>
                    <td>{{ $dataset['label'] }}</td>
                    @foreach ($dataset['data'] as $data)
                        <td>{{ $data }}</td>
                    @endforeach
                    <td>{{ max($dataset['data']) }}</td> <!-- 显示每一行数据的最大值 -->
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
