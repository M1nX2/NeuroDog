<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Storage;
use App\Models\Violation;
use Carbon\Carbon;

class VideoController extends Controller
{
    /**
     * URL Python API
     */
    private $apiUrl;

    public function __construct()
    {
        // В Docker используем имя сервиса, локально можно указать localhost:8000
        $this->apiUrl = env('PYTHON_API_URL', 'http://python-backend:8000');
    }

    /**
     * Проверка доступности Python API
     */
    private function checkApiHealth()
    {
        try {
            $response = Http::timeout(5)->get($this->apiUrl . '/health');
            return $response->successful();
        } catch (\Exception $e) {
            \Log::error('Python API недоступен: ' . $e->getMessage());
            return false;
        }
    }

    /**
     * Загрузка и обработка видео
     */
    public function upload(Request $request)
    {
        // Проверяем доступность API перед обработкой
        if (!$this->checkApiHealth()) {
            return response()->json([
                'success' => false,
                'message' => 'Python API недоступен. Проверьте, что сервис python-backend запущен.'
            ], 503);
        }

        // Проверяем размер файла до валидации
        if ($request->hasFile('videoFile')) {
            $file = $request->file('videoFile');
            $fileSize = $file->getSize();
            $maxSize = 500 * 1024 * 1024; // 500MB в байтах
            
            if ($fileSize > $maxSize) {
                return response()->json([
                    'success' => false,
                    'message' => 'Размер файла (' . round($fileSize / 1024 / 1024, 2) . 'MB) превышает максимально допустимый (500MB)'
                ], 413);
            }
            
            // Проверяем текущие настройки PHP
            $uploadMax = ini_get('upload_max_filesize');
            $postMax = ini_get('post_max_size');
            
            \Log::info("PHP настройки: upload_max_filesize={$uploadMax}, post_max_size={$postMax}, file_size=" . round($fileSize / 1024 / 1024, 2) . "MB");
        }

        try {
            $request->validate([
                'videoFile' => 'required|file|mimes:mp4,avi,mov|max:512000' // 500MB max в KB
            ]);
        } catch (\Illuminate\Validation\ValidationException $e) {
            return response()->json([
                'success' => false,
                'message' => 'Ошибка валидации',
                'errors' => $e->errors()
            ], 422);
        }

        try {
            $file = $request->file('videoFile');
            
            \Log::info('Начало обработки видео: ' . $file->getClientOriginalName());
            
            // Отправляем файл в Python API
            $response = Http::timeout(600) // 10 минут таймаут
                ->attach('file', file_get_contents($file->getRealPath()), $file->getClientOriginalName())
                ->post($this->apiUrl . '/api/v1/process-video');
            
            \Log::info('Ответ от Python API: ' . $response->status());

            if ($response->successful()) {
                $data = $response->json();
                
                // Сохраняем нарушения в базу данных
                if (isset($data['violations']) && is_array($data['violations']) && count($data['violations']) > 0) {
                    try {
                        foreach ($data['violations'] as $violationData) {
                            Violation::create([
                                'time' => $violationData['time'] ?? '00:00:00',
                                'type' => $violationData['type'] ?? 'Неубранные экскременты',
                                'description' => $violationData['description'] ?? '',
                                'source' => $violationData['source'] ?? '',
                                'date' => $violationData['date'] ?? date('Y-m-d'),
                                'video_id' => $data['video_id'] ?? null,
                                'video_url' => $violationData['video_url'] ?? null,
                                'breed' => $violationData['breed'] ?? null,
                                'muzzle' => $violationData['muzzle'] ?? null,
                            ]);
                        }
                        \Log::info('Сохранено нарушений в БД: ' . count($data['violations']));
                    } catch (\Exception $e) {
                        \Log::error('Ошибка сохранения нарушений в БД: ' . $e->getMessage());
                        // Продолжаем выполнение, даже если не удалось сохранить в БД
                    }
                } else {
                    \Log::info('Нарушений не обнаружено, пропускаем сохранение в БД');
                }

                return response()->json([
                    'success' => true,
                    'data' => $data
                ]);
            } else {
                \Log::error('Ошибка Python API: ' . $response->body());
                return response()->json([
                    'success' => false,
                    'message' => 'Ошибка обработки видео: ' . $response->body(),
                    'status' => $response->status()
                ], $response->status() ?: 500);
            }
        } catch (\Illuminate\Http\Client\ConnectionException $e) {
            \Log::error('Ошибка подключения к Python API: ' . $e->getMessage());
            return response()->json([
                'success' => false,
                'message' => 'Не удалось подключиться к Python API. Проверьте, что сервис запущен и доступен по адресу: ' . $this->apiUrl
            ], 503);
        } catch (\Exception $e) {
            \Log::error('Ошибка обработки видео: ' . $e->getMessage());
            \Log::error('Stack trace: ' . $e->getTraceAsString());
            return response()->json([
                'success' => false,
                'message' => 'Ошибка: ' . $e->getMessage()
            ], 500);
        }
    }

    /**
     * Получение списка нарушений
     */
    public function getViolations(Request $request)
    {
        $startDate = $request->input('start_date');
        $endDate = $request->input('end_date');

        try {
            $params = [];
            if ($startDate) {
                $params['start_date'] = $startDate;
            }
            if ($endDate) {
                $params['end_date'] = $endDate;
            }

            $response = Http::get($this->apiUrl . '/api/v1/violations', $params);

            if ($response->successful()) {
                return response()->json([
                    'success' => true,
                    'violations' => $response->json()
                ]);
            } else {
                // Если API недоступен, берем из базы данных
                $query = Violation::query();
                
                if ($startDate) {
                    $query->where('date', '>=', $startDate);
                }
                if ($endDate) {
                    $query->where('date', '<=', $endDate);
                }
                
                $violations = $query->orderBy('date')->orderBy('time')->get();
                
                return response()->json([
                    'success' => true,
                    'violations' => $violations->map(function($v) {
                        return [
                            'time' => $v->time,
                            'type' => $v->type,
                            'description' => $v->description,
                            'source' => $v->source,
                            'date' => $v->date,
                            'video_url' => $v->video_url,
                            'breed' => $v->breed,
                            'muzzle' => $v->muzzle,
                        ];
                    })
                ]);
            }
        } catch (\Exception $e) {
            // Fallback на базу данных
            $query = Violation::query();
            
            if ($startDate) {
                $query->where('date', '>=', $startDate);
            }
            if ($endDate) {
                $query->where('date', '<=', $endDate);
            }
            
            $violations = $query->orderBy('date')->orderBy('time')->get();
            
            return response()->json([
                'success' => true,
                'violations' => $violations->map(function($v) {
                    return [
                        'time' => $v->time,
                        'type' => $v->type,
                        'description' => $v->description,
                        'source' => $v->source,
                        'date' => $v->date,
                        'video_url' => $v->video_url,
                        'breed' => $v->breed,
                        'muzzle' => $v->muzzle,
                    ];
                })
            ]);
        }
    }

    /**
     * Получение прогресса обработки видео
     */
    public function getProgress($videoId)
    {
        try {
            $response = Http::get($this->apiUrl . '/api/v1/progress/' . $videoId);
            
            if ($response->successful()) {
                return response()->json($response->json());
            } else {
                // Если прогресс не найден, возможно обработка завершена
                return response()->json([
                    'percent' => 100,
                    'status' => 'Завершено',
                    'completed' => true
                ]);
            }
        } catch (\Exception $e) {
            return response()->json([
                'percent' => 0,
                'status' => 'Ошибка получения прогресса',
                'completed' => false,
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Получение нарушений для конкретного видео
     */
    public function getVideoViolations($videoId)
    {
        try {
            $response = Http::get($this->apiUrl . '/api/v1/violations/' . $videoId);
            
            if ($response->successful()) {
                $data = $response->json();
                return response()->json([
                    'success' => true,
                    'data' => $data
                ]);
            } else {
                // Если API недоступен, берем из базы данных
                $violations = Violation::where('video_id', $videoId)->get();
                return response()->json([
                    'success' => true,
                    'data' => [
                        'video_id' => $videoId,
                        'violations' => $violations->map(function($v) {
                            return [
                                'time' => $v->time,
                                'type' => $v->type,
                                'description' => $v->description,
                                'source' => $v->source,
                                'date' => $v->date,
                                'video_url' => $v->video_url,
                                'breed' => $v->breed,
                                'muzzle' => $v->muzzle,
                            ];
                        })->toArray(),
                        'processing_time' => null
                    ]
                ]);
            }
        } catch (\Exception $e) {
            // Fallback на базу данных
            $violations = Violation::where('video_id', $videoId)->get();
            return response()->json([
                'success' => true,
                'data' => [
                    'video_id' => $videoId,
                    'violations' => $violations->map(function($v) {
                        return [
                            'time' => $v->time,
                            'type' => $v->type,
                            'description' => $v->description,
                            'source' => $v->source,
                            'date' => $v->date,
                            'video_url' => $v->video_url,
                            'breed' => $v->breed,
                            'muzzle' => $v->muzzle,
                        ];
                    })->toArray(),
                    'processing_time' => null
                ]
            ]);
        }
    }

    /**
     * Получение обработанного видео
     */
    public function getVideo($videoId)
    {
        try {
            $response = Http::get($this->apiUrl . '/api/v1/video/' . $videoId);
            
            if ($response->successful()) {
                return response($response->body(), 200)
                    ->header('Content-Type', 'video/mp4');
            } else {
                return response()->json([
                    'success' => false,
                    'message' => 'Видео не найдено'
                ], 404);
            }
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'message' => 'Ошибка: ' . $e->getMessage()
            ], 500);
        }
    }
}

