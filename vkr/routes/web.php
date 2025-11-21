<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\MainController;
use App\Http\Controllers\ReportController;
use App\Http\Controllers\VideoController;


/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider and all of them will
| be assigned to the "web" middleware group. Make something great!
|
*/

Route::get('/', [MainController::class, 'index']);

// API маршруты для работы с видео
Route::post('/api/video/upload', [VideoController::class, 'upload'])->name('video.upload');
Route::get('/api/violations', [VideoController::class, 'getViolations'])->name('violations.get');
Route::get('/api/violations/{videoId}', [VideoController::class, 'getVideoViolations'])->name('violations.video');
Route::get('/api/video/progress/{videoId}', [VideoController::class, 'getProgress'])->name('video.progress');
Route::get('/api/video/{videoId}', [VideoController::class, 'getVideo'])->name('video.get');

// Отчеты
Route::match(['GET', 'POST'], '/report', [ReportController::class, 'index'])->name('report');
Route::get('/report/export', [ReportController::class, 'exportExcel'])->name('export.excel');
Route::get('/export-excel', [ReportController::class, 'exportExcel'])->name('export.excel');