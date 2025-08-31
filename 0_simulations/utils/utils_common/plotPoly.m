% plot polygon
%%ELiiiiiii, 20211115
function h = plotPoly(polygon)
%poly: N*2 matrix

h = gca;
hold on;
pgon = polyshape(polygon(:,1),polygon(:,2));
plot(pgon,'FaceColor','k','FaceAlpha',0,'LineStyle','--');
hold off;
end