<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Data extends Model
{
    protected $fillable = [
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
}
