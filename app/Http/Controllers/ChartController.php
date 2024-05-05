<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Schema;

use DB;

class ChartController extends Controller
{
    public function LineChart()
    {
        $data = [];
        for ($week = 1; $week <= 10; $week++) {
            $tableName = 'test_' . $week . '_report';

            if (Schema::hasTable($tableName)) {
                $result = DB::select("SELECT Machine, CIavg FROM $tableName");

                foreach ($result as $row) {

                    $machine = $row->Machine;
                    $ciavg = floatval($row->CIavg);
                    if (!isset($data[$machine])) {
                        $data[$machine] = [];
                    }

                    $data[$machine][$week] = $ciavg;
                }
            }
        }

        $labels = array_keys(reset($data));
        $datasets = [];
        foreach ($data as $machine => $values) {
            $dataset = [
                'label' => $machine,
                'data' => array_values($values),
                'borderColor' => '#' . substr(md5($machine), 0, 6),
                'fill' => false
            ];
            $datasets[] = $dataset;
        }

        return view('line-chart', compact('labels', 'datasets'));
    }
}
