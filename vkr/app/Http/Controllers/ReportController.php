<?php

namespace App\Http\Controllers;

use Maatwebsite\Excel\Facades\Excel;
use App\Exports\ReportExport;
use App\Models\Violation;
use Illuminate\Http\Request;
use Carbon\Carbon;

class ReportController extends Controller
{

    public function index(Request $request)
    {
        // Если запрос POST - сохраняем даты в сессию и редиректим
        if ($request->isMethod('post')) {
            $request->validate([
                'start_date' => 'required|date',
                'end_date' => 'required|date|after_or_equal:start_date'
            ]);

            session([
                'report_start_date' => $request->start_date,
                'report_end_date' => $request->end_date
            ]);

            return redirect()->route('report');
        }

        // Для GET запроса берем даты из сессии или по умолчанию
        $startDate = session('report_start_date', Carbon::now()->subMonth()->format('Y-m-d'));
        $endDate = session('report_end_date', Carbon::now()->format('Y-m-d'));

        // Получаем нарушения из базы данных
        $query = Violation::query()
            ->where('date', '>=', $startDate)
            ->where('date', '<=', $endDate)
            ->orderBy('date')
            ->orderBy('time');

        $violations = $query->get()->map(function($violation) {
            return [
                'time' => $violation->time,
                'type' => $violation->type,
                'description' => $violation->description,
                'breed' => $violation->breed,
                'muzzle' => $violation->muzzle,
                'video_url' => $violation->video_url,
                'source' => $violation->source,
                'date' => $violation->date->format('Y-m-d')
            ];
        })->toArray();

        return view('report', [
            'violations' => $violations,
            'startDate' => $startDate,
            'endDate' => $endDate,
            'reportTitle' => "Отчет о нарушениях за период с $startDate по $endDate"
        ]);
    }

    public function exportExcel(Request $request)
    {
        $startDate = session('report_start_date', Carbon::now()->subMonth()->format('Y-m-d'));
        $endDate = session('report_end_date', Carbon::now()->format('Y-m-d'));

        // Получаем нарушения из базы данных
        $violations = Violation::query()
            ->where('date', '>=', $startDate)
            ->where('date', '<=', $endDate)
            ->orderBy('date')
            ->orderBy('time')
            ->get();

        $excelData = $violations->map(function($violation) {
            return [
                'Дата' => $violation->date->format('Y-m-d'),
                'Таймкод' => $violation->time,
                'Тип нарушения' => $violation->type,
                'Описание' => $violation->description,
                'Порода' => $violation->breed ?? 'Не указана',
                'Намордник' => $violation->muzzle ? 'Да' : ($violation->muzzle === null ? 'Не указано' : 'Нет'),
                'Источник видео' => $violation->source,
                'Ссылка на видео' => $violation->video_url ?? 'Не указана'
            ];
        })->toArray();

        $fileName = 'violations_report_' . $startDate . '_to_' . $endDate . '.xlsx';

        return Excel::download(new ReportExport($excelData), $fileName);
    }
}