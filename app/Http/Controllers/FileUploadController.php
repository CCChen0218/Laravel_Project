<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;

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

        $command = escapeshellcmd("$pythonPath $scriptPath");

        $output = shell_exec($command);

        if (!is_null($output)) {
            Log::info($output);
            return response()->json(['message' => 'CSV file processed successfully']);
        } else {;
            return response()->json(['message' => 'No output']);
        }
    }
}


