<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Violation extends Model
{
    use HasFactory;

    protected $fillable = [
        'time',
        'type',
        'description',
        'source',
        'date',
        'video_id',
        'video_url',
        'breed',
        'muzzle',
    ];

    protected $casts = [
        'muzzle' => 'boolean',
        'date' => 'date',
    ];
}

