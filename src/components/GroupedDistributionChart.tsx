import React, {useEffect, useRef} from "react";

export const GroupedDistributionChart = ({
                                      title,
                                      categories,
                                      affected,
                                      unaffected,
                                      height = 200
                                  }: {
    title: string;
    categories: Record<string, number>;  // e.g., gender data
    affected: Record<string, number>;
    unaffected: Record<string, number>;
    height?: number;
}) => {
    const chartRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!chartRef.current) return;

        const labels = Object.keys(categories);
        const affectedValues = labels.map(cat => affected[cat] || 0);
        const unaffectedValues = labels.map(cat => unaffected[cat] || 0);

        const plotData = [
            {
                x: labels,
                y: affectedValues,
                name: 'Affected',
                type: 'bar',
                marker: {color: '#0d6efd'},
                hovertemplate: 'Affected: %{y}<extra></extra>'
            },
            {
                x: labels,
                y: unaffectedValues,
                name: 'Unaffected',
                type: 'bar',
                marker: {color: '#6c757d'},
                hovertemplate: 'Unaffected: %{y}<extra></extra>'
            }
        ];

        const layout = {
            title: {
                text: title,
                font: {size: 12, color: '#666'}
            },
            barmode: 'group',  // Side-by-side bars
            margin: {t: 40, b: 60, l: 40, r: 10},
            height,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            showlegend: true,
            legend: {
                orientation: 'h',
                y: -0.75,
                x: 0.5,
                xanchor: 'center',
                yanchor: 'top',
            },
            xaxis: {
                tickangle: -45,
                tickfont: {size: 10}
            },
            yaxis: {
                title: 'Count',
                gridcolor: '#eee',
                tickfont: {size: 10}
            },
            font: {size: 10}
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        window.Plotly.newPlot(chartRef.current, plotData, layout, config);

        return () => {
            if (chartRef.current) {
                window.Plotly.purge(chartRef.current);
            }
        };
    }, [categories, affected, unaffected, title, height]);

    return <div ref={chartRef} className="w-100"/>;
};