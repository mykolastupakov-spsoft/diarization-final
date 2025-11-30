/**
 * Visualization components for diarization results
 */

class TimelineVisualization {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.services = [];
    this.colors = {
      agreement: '#10B981',
      majority: '#FBBF24',
      dispute: '#EF4444',
      overlap: '#8B5CF6',
      silence: '#9CA3AF',
      speaker_0: '#3B82F6',
      speaker_1: '#F59E0B',
      speaker_2: '#8B5CF6',
      speaker_3: '#10B981'
    };
  }

  /**
   * Render stacked timeline for multiple services
   */
  renderStackedTimeline(servicesData, reference = null) {
    if (!this.container) return;

    const maxTime = Math.max(
      ...servicesData.flatMap(s => s.segments.map(seg => seg.end))
    );

    let html = '<div class="error-timeline">';

    // Add reference if available
    if (reference) {
      html += this.renderTimelineRow('Reference (Ground Truth)', reference.segments, maxTime, true);
    }

    // Add each service
    servicesData.forEach(service => {
      html += this.renderTimelineRow(service.name, service.segments, maxTime, false);
    });

    html += '</div>';
    this.container.innerHTML = html;
  }

  /**
   * Render a single timeline row
   */
  renderTimelineRow(label, segments, maxTime, isReference) {
    const pixelsPerSecond = 10; // Adjust for zoom level
    const width = maxTime * pixelsPerSecond;

    let html = `
      <div class="timeline-row">
        <div class="timeline-label">${label}</div>
        <div class="timeline-bar" style="width: ${width}px; min-width: 100%;">
    `;

    segments.forEach((seg, idx) => {
      const left = (seg.start / maxTime) * 100;
      const segWidth = ((seg.end - seg.start) / maxTime) * 100;
      const color = this.getSpeakerColor(seg.speaker, isReference);
      
      html += `
        <div class="timeline-segment" 
             style="position: absolute; left: ${left}%; width: ${segWidth}%; background: ${color};" 
             title="${seg.speaker}: ${seg.start.toFixed(1)}s - ${seg.end.toFixed(1)}s">
        </div>
      `;
    });

    html += '</div></div>';
    return html;
  }

  /**
   * Get color for speaker
   */
  getSpeakerColor(speaker, isReference = false) {
    // Extract speaker number
    const match = speaker.match(/\d+/);
    const speakerNum = match ? parseInt(match[0]) : 0;
    const colorKey = `speaker_${speakerNum % 4}`;
    
    let color = this.colors[colorKey] || '#3B82F6';
    
    // Make reference slightly transparent
    if (isReference) {
      return color + 'CC';
    }
    
    return color;
  }

  /**
   * Highlight error segments
   */
  highlightErrors(errors) {
    // Add error overlays to timeline
    const errorColors = {
      fa: this.colors.dispute,
      miss: this.colors.majority,
      conf: this.colors.overlap
    };

    errors.forEach(error => {
      // Add visual indicator for error
      console.log(`Error at ${error.start}-${error.end}: ${error.type}`);
    });
  }
}

class MetricsChart {
  /**
   * Create comparison bar chart
   */
  static renderComparisonBars(containerId, data, metricName) {
    const container = document.getElementById(containerId);
    if (!container) return;

    let html = '<div class="comparison-bars">';

    data.forEach(item => {
      const value = parseFloat(item.value);
      const width = Math.min(100, value);
      const colorClass = value < 10 ? 'good' : value < 20 ? 'acceptable' : 'poor';

      html += `
        <div class="comparison-bar">
          <div class="comparison-bar-label">
            <strong>${item.name}</strong>
            <span class="metric-value ${colorClass}">${item.value}${item.unit || '%'}</span>
          </div>
          <div class="comparison-bar-track">
            <div class="comparison-bar-fill" style="width: ${width}%; background: ${item.color || 'var(--color-primary)'};"></div>
          </div>
        </div>
      `;
    });

    html += '</div>';
    container.innerHTML = html;
  }

  /**
   * Render metrics grid
   */
  static renderMetricsGrid(containerId, metrics) {
    const container = document.getElementById(containerId);
    if (!container) return;

    let html = '<div class="metrics-grid">';

    Object.entries(metrics).forEach(([key, data]) => {
      const valueClass = data.quality || '';
      html += `
        <div class="metric-card">
          <div class="metric-label">
            ${data.icon || '📊'} ${data.label}
            ${data.info ? `<span class="tooltip-trigger" title="${data.info}">ℹ️</span>` : ''}
          </div>
          <div class="metric-value ${valueClass}">${data.value}</div>
          ${data.subtitle ? `<div class="metric-subtitle">${data.subtitle}</div>` : ''}
        </div>
      `;
    });

    html += '</div>';
    container.innerHTML = html;
  }
}

class AgreementMatrix {
  /**
   * Render agreement matrix table
   */
  static render(containerId, matrix, serviceNames) {
    const container = document.getElementById(containerId);
    if (!container) return;

    let html = '<div class="agreement-matrix"><table class="agreement-table"><thead><tr><th></th>';

    // Header row
    Object.keys(matrix).forEach(serviceId => {
      const name = serviceNames[serviceId] || serviceId;
      html += `<th>${name}</th>`;
    });
    html += '</tr></thead><tbody>';

    // Data rows
    Object.entries(matrix).forEach(([serviceIdA, row]) => {
      const nameA = serviceNames[serviceIdA] || serviceIdA;
      html += `<tr><th>${nameA}</th>`;
      
      Object.entries(row).forEach(([serviceIdB, agreement]) => {
        const value = parseFloat(agreement);
        let cellClass = 'agreement-cell ';
        
        if (value >= 95) cellClass += 'agreement-high';
        else if (value >= 85) cellClass += 'agreement-medium';
        else cellClass += 'agreement-low';

        html += `<td class="${cellClass}">${agreement}%</td>`;
      });
      
      html += '</tr>';
    });

    html += '</tbody></table></div>';
    container.innerHTML = html;
  }
}