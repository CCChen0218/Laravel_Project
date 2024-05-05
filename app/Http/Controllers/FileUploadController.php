<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\Log;
use App\Models\Data;
use DB;

class FileUploadController extends Controller
{
    public function showUploadForm()
    {
        return view('upload_form');
    }

    public function upload(Request $request)
    {
        if ($request->hasFile('file')) {
            $file = $request->file('file');

            $FilePath = $file->move(public_path('uploads'), $file->getClientOriginalName());

            return response()->json(['message' => 'CSV file uploaded successfully', 'tempFilePath' => $FilePath]);
        }

        return response()->json(['error' => 'Please select a file to upload'], 400);
    }

    public function process()
    {
        $pythonPath = 'C:\\Users\\cjbor\\AppData\\Local\\Programs\\Python\\Python39\\python.exe';
        $scriptPath = public_path('uploads/EM_v2_bootstrap.py');
        $csvFilePath = glob(public_path('uploads/test_[1-9]_report.csv'));

        Log::info($csvFilePath);

        $command = escapeshellcmd("$pythonPath $scriptPath");

        $output = shell_exec($command);

        foreach ($csvFilePath as $csvFile) {
            if (!is_null($output)) {
                Log::info($output);
    
                if (file_exists($csvFile)) {

                    $csvData = array_map('str_getcsv', file($csvFile));
    
                    $originalFileName = pathinfo($csvFile, PATHINFO_FILENAME);
    
                    $tableName = str_replace(['.', '/', '\\'], '_', $originalFileName);
    
                    $headers = [
                        'Machine',
                        'machine_bad_pieces',
                        'good_produced',
                        'init_miss_rate',
                        'init_yieldrate',
                        'YieldRate',
                        'Missrate',
                        'ln_yieldrate',
                        'ln_missrate',
                        'prev_yieldrate',
                        'BadPiece',
                        'exp_pieces_going_bad',
                        'not_detected_bad_pieces',
                        'exp_num_good_pieces',
                        'sum_exp',
                        'sum_miss',
                        'CIlow',
                        'CIhigh',
                        'CIavg',
                        'CIwidth',
                        'ProcessedPieces',
                        'NumBatches'
                    ];
    
                    Log::info('Headers:', $headers);
                
                    if (Schema::hasTable($tableName)) {
                        DB::table($tableName)->truncate();
                    } else {
                        $columns = implode(', ', array_map(function($header) {
                            return "$header VARCHAR(255)";
                        }, $headers));
                        $sqlCreate = "CREATE TABLE `$tableName` ($columns)";
                        DB::statement($sqlCreate);
                    }
    
                    foreach ($csvData as $index => $row) {

                        if ($index === 0) {
                            continue;
                        }
                    
                        Log::info('Row:', $row);

                        $placeholders = array_fill(0, count($headers), '?');
                        
                        $values = implode(", ", $placeholders);
                        
                        $sql = "INSERT INTO `$tableName` VALUES ($values)";
                        
                        DB::statement($sql, $row);
                    }

                } else {
                    return response()->json(['error' => 'CSV file not found']);
                }
            } else {
                return response()->json(['message' => 'No output']);
            }
        }
        return response()->json(['message' => 'CSV file processed and imported successfully']);        
    }
}


