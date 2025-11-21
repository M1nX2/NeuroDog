<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('violations', function (Blueprint $table) {
            $table->id();
            $table->string('time'); // Время нарушения в формате HH:MM:SS
            $table->string('type'); // Тип нарушения
            $table->text('description'); // Описание
            $table->string('source'); // Имя исходного файла
            $table->date('date'); // Дата нарушения
            $table->string('video_id')->nullable(); // ID обработанного видео
            $table->string('video_url')->nullable(); // URL обработанного видео
            $table->string('breed')->nullable(); // Порода собаки
            $table->boolean('muzzle')->nullable(); // Наличие намордника
            $table->timestamps();
            
            $table->index(['date', 'time']);
            $table->index('video_id');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('violations');
    }
};

