<?php

namespace App\Exports;

use Maatwebsite\Excel\Concerns\FromArray;
use Maatwebsite\Excel\Concerns\WithHeadings;
use Maatwebsite\Excel\Concerns\WithStyles;
use Maatwebsite\Excel\Concerns\WithColumnWidths;
use PhpOffice\PhpSpreadsheet\Worksheet\Worksheet;

class ReportExport implements FromArray, WithHeadings, WithStyles, WithColumnWidths
{
    protected $data;

    public function __construct(array $data)
    {
        $this->data = $data;
    }

    public function array(): array
    {
        return $this->data;
    }

    public function headings(): array
    {
        return [
            'Дата',
            'Тип нарушения',
            'Таймкод',
            'Описание',
            'Порода',
            'Намордник',
            'Источник видео',
            'Ссылка на видео'
        ];
    }

    public function styles(Worksheet $sheet)
    {
        return [
            1 => ['font' => ['bold' => true]],
            'G' => ['alignment' => ['wrapText' => true]],
            'H' => ['alignment' => ['wrapText' => true]],
        ];
    }

    public function columnWidths(): array
    {
        return [
            'A' => 12, // Дата
            'B' => 12, // Таймкод
            'C' => 25, // Тип нарушения
            'D' => 40, // Описание
            'E' => 20, // Порода
            'F' => 15, // Намордник
            'G' => 20, // Источник видео
            'H' => 50, // Ссылка на видео
        ];
    }
}