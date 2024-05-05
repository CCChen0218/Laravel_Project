<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateCsvDataTable extends Migration
{
    public function up()
    {
        Schema::create('csv_data', function (Blueprint $table) {
            $table->id();
            $table->string('Machine');
            $table->string('machine_bad_pieces');
            $table->string('init_miss_rate');
            $table->string('init_yieldrate');
            $table->string('YieldRate');
            $table->string('Missrate');
            $table->string('ln_yieldrate');
            $table->string('ln_missrate');
            $table->string('prev_yieldrate');
            $table->string('BadPiece');
            $table->string('exp_pieces_going_bad');
            $table->string('not_detected_bad_pieces');
            $table->string('exp_num_good_pieces');
            $table->string('sum_exp');
            $table->string('sum_miss');
            $table->string('CIlow');
            $table->string('CIhigh');
            $table->string('CIavg');
            $table->string('CIwidth');
            $table->string('ProcessedPieces');
            $table->string('NumBatches');
            $table->timestamps();
        });
    }

    public function down()
    {
        Schema::dropIfExists('csv_data');
    }
}
