<?php

namespace App\Http\Controllers;

use App\Models\Data;
use Illuminate\Http\Request;

class CsvController extends Controller
{
    public function import(Request $request)
    {
        $file = $request->file('csv_file');

        $csvData = array_map('str_getcsv', file($file));

        $firstRowSkipped = false;

        foreach ($csvData as $data) {

            if (!$firstRowSkipped) {
                $firstRowSkipped = true;
                continue;
            }

            CsvData::create([
                'Machine' => $data[0],
                'machine_bad_pieces' => $data[1],
                'init_miss_rate' => $data[2],
                'init_yieldrate' => $data[3],
                'YieldRate' => $data[4],
                'Missrate' => $data[5],
                'ln_yieldrate' => $data[6],
                'ln_missrate' => $data[7],
                'prev_yieldrate' => $data[8],
                'BadPiece' => $data[9],
                'exp_pieces_going_bad' => $data[10],
                'not_detected_bad_pieces' => $data[11],
                'exp_num_good_pieces' => $data[12],
                'sum_exp' => $data[13],
                'sum_miss' => $data[14],
                'CIlow' => $data[15],
                'CIhigh' => $data[16],
                'CIavg' => $data[17],
                'CIwidth' => $data[18],
                'ProcessedPieces' => $data[19],
                'NumBatches' => $data[20]
            ]);
        }

        return redirect()->back()->with('success', 'CSV data imported successfully.');
    }
}
